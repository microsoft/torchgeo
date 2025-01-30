# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EnMAP dataset."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import RGBBandsMissingError
from .geo import RasterDataset
from .utils import Path, percentile_normalization

ALL_BANDS = list(range(1, 225))
# Remove bands strongly affected by water vapor absorption due to presence of nodata:
# * https://arxiv.org/abs/2306.00385
# * https://arxiv.org/abs/2408.08447
DEFAULT_BANDS = list(range(1, 127)) + list(range(142, 161)) + list(range(168, 225))


class EnMAP(RasterDataset):
    """`EnMAP <https://www.enmap.org/>`__ dataset.

    The Environmental Mapping and Analysis Program (EnMAP) is a German hyperspectral
    satellite mission that monitors and characterizes Earth's environment on a global
    scale. EnMAP measures geochemical, biochemical and biophysical variables providing
    information on the status and evolution of terrestrial and aquatic ecosystems.

    Mission Outline:

    * Dedicated pushbroom hyperspectral imager mainly based on modified existing or
      pre-developed technology
    * Broad spectral range from 420 nm to 1000 nm (VNIR) and from 900 nm to 2450 nm
      (SWIR) with high radiometric resolution and stability in both spectral ranges
    * 30 km swath width at a spatial resolution of 30 x 30 m, nadir revisit time of
      27 days and off-nadir (30°) pointing feature for fast target revisit (4 days)
    * Sufficient on-board memory to acquire 1,000 km swath length per orbit and a
      total of 5,000 km per day.

    If you use this dataset in your research, please cite the following papers:

    * https://doi.org/10.1016/j.rse.2024.114379
    * https://doi.org/10.1016/j.rse.2023.113632

    .. versionadded:: 0.7
    """

    # https://www.enmap.org/data/doc/EN-PCV-ICD-2009-2_HSI_Product_Specification_Level1_Level2.pdf
    filename_glob = 'ENMAP*SPECTRAL_IMAGE*'
    filename_regex = r"""
        ^ENMAP
        (?P<satellite>\d{2})-
        (?P<product_type>____L[12][ABC])-
        (?P<datatake_id>DT\d{10})_
        (?P<date>\d{8}T\d{6})Z_
        (?P<tile_id>\d{3})_
        (?P<version>V\d{6})_
        (?P<processing_date>\d{8}T\d{6})Z-
    """
    date_format = '%Y%m%dT%H%M%S'

    all_bands = tuple(f'B{n}' for n in ALL_BANDS)
    default_bands = tuple(f'B{n}' for n in DEFAULT_BANDS)
    rgb_bands = ('B48', 'B30', 'B16')

    # Exact values vary from image to image, here are useful defaults:
    # https://www.enmap.org/data/doc/EnMAP_Spectral_Bands_update.xlsx
    wavelengths: ClassVar[dict[str, float]] = {
        'B1': 0.418416,
        'B2': 0.424043,
        'B3': 0.429457,
        'B4': 0.434686,
        'B5': 0.439758,
        'B6': 0.444699,
        'B7': 0.449539,
        'B8': 0.454306,
        'B9': 0.459031,
        'B10': 0.463730,
        'B11': 0.468411,
        'B12': 0.473080,
        'B13': 0.477744,
        'B14': 0.482411,
        'B15': 0.487087,
        'B16': 0.491780,
        'B17': 0.496497,
        'B18': 0.501243,
        'B19': 0.506020,
        'B20': 0.510829,
        'B21': 0.515672,
        'B22': 0.520551,
        'B23': 0.525467,
        'B24': 0.530424,
        'B25': 0.535422,
        'B26': 0.540463,
        'B27': 0.545551,
        'B28': 0.550687,
        'B29': 0.555873,
        'B30': 0.561112,
        'B31': 0.566405,
        'B32': 0.571756,
        'B33': 0.577166,
        'B34': 0.582636,
        'B35': 0.588171,
        'B36': 0.593773,
        'B37': 0.599446,
        'B38': 0.605193,
        'B39': 0.611017,
        'B40': 0.616923,
        'B41': 0.622921,
        'B42': 0.628987,
        'B43': 0.635112,
        'B44': 0.641294,
        'B45': 0.647537,
        'B46': 0.653841,
        'B47': 0.660207,
        'B48': 0.666637,
        'B49': 0.673131,
        'B50': 0.679691,
        'B51': 0.686319,
        'B52': 0.693014,
        'B53': 0.699780,
        'B54': 0.706617,
        'B55': 0.713524,
        'B56': 0.720501,
        'B57': 0.727545,
        'B58': 0.734654,
        'B59': 0.741826,
        'B60': 0.749060,
        'B61': 0.756353,
        'B62': 0.763703,
        'B63': 0.771108,
        'B64': 0.778567,
        'B65': 0.786078,
        'B66': 0.793639,
        'B67': 0.801249,
        'B68': 0.808905,
        'B69': 0.816608,
        'B70': 0.824355,
        'B71': 0.832145,
        'B72': 0.839976,
        'B73': 0.847847,
        'B74': 0.855757,
        'B75': 0.863703,
        'B76': 0.871683,
        'B77': 0.879693,
        'B78': 0.887729,
        'B79': 0.895789,
        'B80': 0.903870,
        'B81': 0.911968,
        'B82': 0.920081,
        'B83': 0.928204,
        'B84': 0.936335,
        'B85': 0.944470,
        'B86': 0.952608,
        'B87': 0.960748,
        'B88': 0.968892,
        'B89': 0.977037,
        'B90': 0.985186,
        'B91': 0.993338,
        'B92': 0.901961,
        'B93': 0.911571,
        'B94': 0.921320,
        'B95': 0.931203,
        'B96': 0.941218,
        'B97': 0.951360,
        'B98': 0.961628,
        'B99': 0.972016,
        'B100': 0.982523,
        'B101': 0.993144,
        'B102': 1.00388,
        'B103': 1.01472,
        'B104': 1.02566,
        'B105': 1.03670,
        'B106': 1.04784,
        'B107': 1.05907,
        'B108': 1.07039,
        'B109': 1.08178,
        'B110': 1.09326,
        'B111': 1.10481,
        'B112': 1.11643,
        'B113': 1.12810,
        'B114': 1.13984,
        'B115': 1.15162,
        'B116': 1.16344,
        'B117': 1.17530,
        'B118': 1.18720,
        'B119': 1.19911,
        'B120': 1.21105,
        'B121': 1.22300,
        'B122': 1.23497,
        'B123': 1.24694,
        'B124': 1.25893,
        'B125': 1.27092,
        'B126': 1.28292,
        'B127': 1.29491,
        'B128': 1.30690,
        'B129': 1.31888,
        'B130': 1.33085,
        'B131': 1.34282,
        'B132': 1.35476,
        'B133': 1.36669,
        'B134': 1.37860,
        'B135': 1.39048,
        'B136': 1.46110,
        'B137': 1.47274,
        'B138': 1.48434,
        'B139': 1.49589,
        'B140': 1.50740,
        'B141': 1.51887,
        'B142': 1.53029,
        'B143': 1.54167,
        'B144': 1.55301,
        'B145': 1.56430,
        'B146': 1.57555,
        'B147': 1.58676,
        'B148': 1.59791,
        'B149': 1.60902,
        'B150': 1.62009,
        'B151': 1.63111,
        'B152': 1.64207,
        'B153': 1.65300,
        'B154': 1.66387,
        'B155': 1.67470,
        'B156': 1.68547,
        'B157': 1.69620,
        'B158': 1.70687,
        'B159': 1.71750,
        'B160': 1.72808,
        'B161': 1.73860,
        'B162': 1.74908,
        'B163': 1.75951,
        'B164': 1.93914,
        'B165': 1.94869,
        'B166': 1.95820,
        'B167': 1.96766,
        'B168': 1.97708,
        'B169': 1.98645,
        'B170': 1.99579,
        'B171': 2.00508,
        'B172': 2.01433,
        'B173': 2.02354,
        'B174': 2.03270,
        'B175': 2.04183,
        'B176': 2.05092,
        'B177': 2.05996,
        'B178': 2.06897,
        'B179': 2.07793,
        'B180': 2.08686,
        'B181': 2.09574,
        'B182': 2.10459,
        'B183': 2.11340,
        'B184': 2.12217,
        'B185': 2.13090,
        'B186': 2.13960,
        'B187': 2.14826,
        'B188': 2.15688,
        'B189': 2.16547,
        'B190': 2.17402,
        'B191': 2.18253,
        'B192': 2.19101,
        'B193': 2.19945,
        'B194': 2.20786,
        'B195': 2.21624,
        'B196': 2.22458,
        'B197': 2.23289,
        'B198': 2.24116,
        'B199': 2.24940,
        'B200': 2.25761,
        'B201': 2.26579,
        'B202': 2.27393,
        'B203': 2.28204,
        'B204': 2.29012,
        'B205': 2.29817,
        'B206': 2.30619,
        'B207': 2.31417,
        'B208': 2.32213,
        'B209': 2.33005,
        'B210': 2.33794,
        'B211': 2.34581,
        'B212': 2.35364,
        'B213': 2.36144,
        'B214': 2.36921,
        'B215': 2.37695,
        'B216': 2.38466,
        'B217': 2.39234,
        'B218': 2.40000,
        'B219': 2.40762,
        'B220': 2.41521,
        'B221': 2.42278,
        'B222': 2.43032,
        'B223': 2.43782,
        'B224': 2.44530,
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new EnMAP instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        bands = bands or self.default_bands
        super().__init__(paths, crs, res, bands, transforms, cache)

    def plot(self, sample: dict[str, Any], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`RasterDataset.__getitem__`.
            suptitle: optional string to use as a suptitle

        Returns:
            A matplotlib Figure with the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].cpu().numpy()
        image = np.clip(image, 0, None)
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        if suptitle:
            fig.suptitle(suptitle)

        return fig
