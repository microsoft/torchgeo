# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Cheasapeake Bay Program Land Use/Land Cover Data Project datasets."""

import functools
import glob
import operator
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset, UnionDataset
from .nlcd import NLCD
from .utils import Path, Sample, download_url, extract_archive


class Chesapeake(RasterDataset, ABC):
    """Abstract base class for all Chesapeake datasets.

    `Chesapeake Bay Land Use and Land Cover (LULC) Database 2022 Edition
    <https://www.chesapeakeconservancy.org/projects/cbp-land-use-land-cover-data-project>`_

    The Chesapeake Bay Land Use and Land Cover Database (LULC) facilitates
    characterization of the landscape and land change for and between discrete time
    periods. The database was developed by the University of Vermont's Spatial Analysis
    Laboratory in cooperation with Chesapeake Conservancy (CC) and U.S. Geological
    Survey (USGS) as part of a 6-year Cooperative Agreement between Chesapeake
    Conservancy and the U.S. Environmental Protection Agency (EPA) and a separate
    Interagency Agreement between the USGS and EPA to provide geospatial support to the
    Chesapeake Bay Program Office.

    The database contains one-meter 13-class Land Cover (LC) and 54-class Land Use/Land
    Cover (LULC) for all counties within or adjacent to the Chesapeake Bay watershed for
    2013/14 and 2017/18, depending on availability of National Agricultural Imagery
    Program (NAIP) imagery for each state. Additionally, 54 LULC classes are generalized
    into 18 LULC classes for ease of visualization and communication of LULC trends. LC
    change between discrete time periods, detected by spectral changes in NAIP imagery
    and LiDAR, represents changes between the 12 land cover classes. LULC change uses LC
    change to identify where changes are happening and then LC is translated to LULC to
    represent transitions between the 54 LULC classes. The LULCC data is represented as
    a LULC class change transition matrix which provides users acres of change between
    multiple classes. It is organized by 18x18 and 54x54 LULC classes. The Chesapeake
    Bay Water (CBW) indicates raster tabulations were performed for only areas that fall
    inside the CBW boundary e.g., if user is interested in CBW portion of a county then
    they will use LULC Matrix CBW. Conversely, if they are interested change transitions
    across the entire county, they will use LULC Matrix.

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/P981GV1L
    """

    url = 'https://hf.co/datasets/torchgeo/chesapeake/resolve/1e0370eda6a24d93af4153745e54fd383d015bf5/{state}_lulc_{year}_2022-Edition.zip'
    filename_glob = '{state}_lulc_*_2022-Edition.tif'
    filename_regex = r'^{state}_lulc_(?P<date>\d{{4}})_2022-Edition\.tif$'
    date_format = '%Y'
    is_image = False

    @property
    @abstractmethod
    def sha256s(self) -> dict[int, str]:
        """Mapping between data year and zip file sha256."""

    @property
    def state(self) -> str:
        """State abbreviation."""
        return self.__class__.__name__[-2:].lower()

    cmap = ListedColormap(
        np.array(
            [
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (0, 0, 0, 255),
                (235, 6, 2, 255),
                (89, 89, 89, 255),
                (138, 138, 136, 255),
                (138, 138, 136, 255),
                (138, 138, 136, 255),
                (115, 115, 0, 255),
                (233, 255, 190, 255),
                (255, 255, 115, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (38, 115, 0, 255),
                (56, 168, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 115, 255),
                (255, 255, 115, 255),
                (255, 255, 115, 255),
                (170, 255, 0, 255),
                (170, 255, 0, 255),
                (170, 255, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (77, 209, 148, 255),
                (77, 209, 148, 255),
                (56, 168, 0, 255),
                (38, 115, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (186, 245, 217, 255),
                (186, 245, 217, 255),
                (56, 168, 0, 255),
                (38, 115, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 211, 127, 255),
                (255, 211, 127, 255),
                (255, 211, 127, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (0, 168, 132, 255),
                (0, 168, 132, 255),
                (0, 168, 132, 255),
                (56, 168, 0, 255),
                (38, 115, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
            ]
        )
        / 255
    )

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = True,
        time_series: bool = False,
    ) -> None:
        """Initialize a new Chesapeake instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files (may be slow)
            time_series: if True, stack data along the time series dimension
                [T, C, H, W]. If False, merge data into a [C, H, W] mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.9
           The *time_series* parameter.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.filename_glob = self.filename_glob.format(state=self.state)
        self.filename_regex = self.filename_regex.format(state=self.state)

        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted file already exists
        if self.files:
            return

        # Check if the zip file has already been downloaded
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        if glob.glob(os.path.join(paths, '**', '*.zip'), recursive=True):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        for year, sha256 in self.sha256s.items():
            url = self.url.format(state=self.state, year=year)
            download_url(url, paths, sha256=sha256 if self.checksum else None)

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        for file in glob.iglob(os.path.join(paths, '**', '*.zip'), recursive=True):
            extract_archive(file)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        mask = sample['mask']
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction']
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(4 * ncols, 4))
        kwargs = {'cmap': self.cmap, 'vmin': 0, 'vmax': 128, 'interpolation': 'none'}

        axs[0, 0].imshow(mask, **kwargs)
        axs[0, 0].axis('off')
        if show_titles:
            axs[0, 0].set_title('Mask')

        if showing_predictions:
            axs[0, 1].imshow(pred, **kwargs)
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class ChesapeakeDC(Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    sha256s: ClassVar[dict[int, str]] = {
        2013: '13c776f8690a19c4088df6f4c143bca9650ce6cfe5c4091f3b774441eab8805a',
        2017: 'f71ae8836bbbf7e4b60e4edf99e3a6eee16f62473dcb2c7f20ba5836c7a108c8',
    }


class ChesapeakeDE(Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    sha256s: ClassVar[dict[int, str]] = {
        2013: 'ced3e274bfd8531915cb21d1a3faad31c9de859648feb8b8daec245f343c0b5c',
        2018: '4b996051cbd532dc4e43642d4ecbdf2dd55456a927edc35983b427f137145273',
    }


class ChesapeakeMD(Chesapeake):
    """This subset of the dataset contains data only for Maryland."""

    sha256s: ClassVar[dict[int, str]] = {
        2013: 'e5a6a2c02f50295f6f13539092ce801c94263bd7ac304203cc2da2d4d774dd18',
        2018: 'd3b70ae09737e119f5292b071f752add4c44c1f0a0b959a26305f2ceb0c583ec',
    }


class ChesapeakeNY(Chesapeake):
    """This subset of the dataset contains data only for New York."""

    sha256s: ClassVar[dict[int, str]] = {
        2013: '2b861e5893f9340fc792c3df2dedfd0230ee35b1cd22fb53cc258fa7a97f4d33',
        2017: 'eceaae776c49636730a51b5abe93ec123000535e92aee37cf9594c8ba52cd41e',
    }


class ChesapeakePA(Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    sha256s: ClassVar[dict[int, str]] = {
        2013: '5f197d18765ff8e0c837433c2eba285f97a1c5ac68c12d8c06fc4854c5df3d8c',
        2017: 'f699ea705ae5bcda04e0eaa5c14369bccc1cec7ad65b1c4946134e8c55b84737',
    }


class ChesapeakeVA(Chesapeake):
    """This subset of the dataset contains data only for Virginia."""

    sha256s: ClassVar[dict[int, str]] = {
        2014: '9e71799f70c4c994eadeb126e4923e7b843dff335b46cb27a4785f1551660226',
        2018: 'c7dd3514268f83fe8a3d650b3e686f4a7bce091b04f6e16a84e4f3d475bf19e9',
    }


class ChesapeakeWV(Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    sha256s: ClassVar[dict[int, str]] = {
        2014: 'a882edc5e71acae0da953e44266562a0a957c8b139a885a20bbd33b22fe43d42',
        2018: '84c3577680bba0c2da179b0def119f014581cdf08d16039188d6e482a997cb8d',
    }


class ChesapeakeCVPRHelper(RasterDataset):
    """This is a helper class for the ChesapeakeCVPR dataset."""

    def __init__(self, paths: str, layer: str, *args: Any, **kwargs: Any) -> None:
        """Initialize helper class.

        Args:
            paths: directory, where dataset is located
            layer: data layer to load
            *args: optional arguments
            **kwargs: optional keyword arguments
        """
        self.filename_glob = f'*_{layer}.tif'
        self.filename_regex = rf'^m_\d+_[a-z]+_\d+_\d+_{layer}\.tif'

        self.is_image = layer in [
            'naip-new',
            'naip-old',
            'landsat-leaf-on',
            'landsat-leaf-off',
        ]
        super().__init__(paths, *args, **kwargs)


class ChesapeakeCVPR(UnionDataset):
    """CVPR 2019 Chesapeake Land Cover dataset.

    The `CVPR 2019 Chesapeake Land Cover
    <https://lila.science/datasets/chesapeakelandcover>`_ dataset contains two layers of
    NAIP aerial imagery, Landsat 8 leaf-on and leaf-off imagery, Chesapeake Bay land
    cover labels, NLCD land cover labels, and Microsoft building footprint labels.

    This dataset was organized to accompany the 2019 CVPR paper, "Large Scale
    High-Resolution Land Cover Mapping with Multi-Resolution Data".

    The paper "Resolving label uncertainty with implicit generative models" added an
    additional layer of data to this dataset containing a prior over the Chesapeake Bay
    land cover classes generated from the NLCD land cover labels. For more information
    about this layer see `the dataset documentation
    <https://zenodo.org/records/5866525>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

    subdatasets: tuple[str, ...] = ('base', 'prior_extension')
    urls: ClassVar[dict[str, str]] = {
        'base': 'https://lilawildlife.blob.core.windows.net/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover.zip',
        'prior_extension': 'https://zenodo.org/records/5866525/files/cvpr_chesapeake_landcover_prior_extension.zip?download=1',
    }
    filenames: ClassVar[dict[str, str]] = {
        'base': 'cvpr_chesapeake_landcover.zip',
        'prior_extension': 'cvpr_chesapeake_landcover_prior_extension.zip',
    }
    md5s: ClassVar[dict[str, str]] = {
        'base': '1225ccbb9590e9396875f221e5031514',
        'prior_extension': '402f41d07823c8faf7ea6960d7c4e17a',
    }

    _res = (1, 1)

    lc_cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 0),
                (0, 197, 255, 255),
                (38, 115, 0, 255),
                (163, 255, 115, 255),
                (255, 170, 0, 255),
                (156, 156, 156, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            ]
        )
        / 255
    )

    prior_color_matrix = np.array(
        [
            [0.0, 0.77254902, 1.0, 1.0],
            [0.14901961, 0.45098039, 0.0, 1.0],
            [0.63921569, 1.0, 0.45098039, 1.0],
            [0.61176471, 0.61176471, 0.61176471, 1.0],
        ]
    )

    valid_layers = (
        'naip-new',
        'naip-old',
        'landsat-leaf-on',
        'landsat-leaf-off',
        'nlcd',
        'lc',
        'buildings',
        'prior_from_cooccurrences_101_31_no_osm_no_buildings',
    )
    states = ('de', 'md', 'va', 'wv', 'pa', 'ny')
    splits = (
        [f'{state}-train' for state in states]
        + [f'{state}-val' for state in states]
        + [f'{state}-test' for state in states]
    )

    # the layer that is only distributed in the prior extension archive
    prior_layer = 'prior_from_cooccurrences_101_31_no_osm_no_buildings'

    # these are used to check the integrity of each subdataset
    _files: ClassVar[dict[str, tuple[str, ...]]] = {
        'base': (
            'de_1m_2013_extended-debuffered-test_tiles',
            'de_1m_2013_extended-debuffered-train_tiles',
            'de_1m_2013_extended-debuffered-val_tiles',
            'md_1m_2013_extended-debuffered-test_tiles',
            'md_1m_2013_extended-debuffered-train_tiles',
            'md_1m_2013_extended-debuffered-val_tiles',
            'ny_1m_2013_extended-debuffered-test_tiles',
            'ny_1m_2013_extended-debuffered-train_tiles',
            'ny_1m_2013_extended-debuffered-val_tiles',
            'pa_1m_2013_extended-debuffered-test_tiles',
            'pa_1m_2013_extended-debuffered-train_tiles',
            'pa_1m_2013_extended-debuffered-val_tiles',
            'va_1m_2014_extended-debuffered-test_tiles',
            'va_1m_2014_extended-debuffered-train_tiles',
            'va_1m_2014_extended-debuffered-val_tiles',
            'wv_1m_2014_extended-debuffered-test_tiles',
            'wv_1m_2014_extended-debuffered-train_tiles',
            'wv_1m_2014_extended-debuffered-val_tiles',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_buildings.tif',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-off.tif',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-on.tif',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_lc.tif',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-new.tif',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-old.tif',
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_nlcd.tif',
            'spatial_index.geojson',
        ),
        'prior_extension': (
            'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif',
        ),
    }

    def __init__(
        self,
        root: Path = 'data',
        splits: Sequence[str] = ['de-train'],
        layers: Sequence[str] = ['naip-new', 'lc'],
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            splits: a list of strings in the format "{state}-{train,val,test}"
                indicating the subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old", "lc", "nlcd",
                "landsat-leaf-on", "landsat-leaf-off", "buildings", or
                "prior_from_cooccurrences_101_31_no_osm_no_buildings" indicating which
                layers to load
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``splits`` or ``layers`` are not valid
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionchanged:: 0.10
           Only the subdatasets required by *layers* are verified and downloaded.
           The prior extension archive is no longer needed unless
           ``"prior_from_cooccurrences_101_31_no_osm_no_buildings"`` is requested.
        """
        assert set(splits) <= set(self.splits)
        assert set(layers) <= set(self.valid_layers)

        self.root = root
        self.layers = layers
        self.transforms = transforms
        self.cache = cache
        self.download = download
        self.checksum = checksum

        if self.prior_layer in layers:
            self.subdatasets = ('base', 'prior_extension')
        else:
            self.subdatasets = ('base',)

        self._verify()

        split_datasets = []
        for split in splits:
            state, split_type = split.split('-')
            directory = os.path.join(self.root, '**', f'{state}_*-{split_type}_tiles')
            directory = glob.glob(directory, recursive=True)[0]

            layer_datasets = []
            for layer in self.layers:
                dataset = ChesapeakeCVPRHelper(directory, layer, cache=self.cache)
                layer_datasets.append(dataset)

            dataset = functools.reduce(operator.and_, layer_datasets)
            split_datasets.append(dataset)

        dataset = functools.reduce(operator.or_, split_datasets)
        self.index = dataset.index
        self.datasets = dataset.datasets
        self.collate_fn = dataset.collate_fn

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""

        def exists(filename: Path) -> bool:
            return os.path.exists(os.path.join(self.root, filename))

        # Check if the extracted files already exist
        if all(
            exists(filename)
            for subdataset in self.subdatasets
            for filename in self._files[subdataset]
        ):
            return

        # Check if the zip files have already been downloaded
        if all(
            os.path.exists(os.path.join(self.root, self.filenames[subdataset]))
            for subdataset in self.subdatasets
        ):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for subdataset in self.subdatasets:
            download_url(
                self.urls[subdataset],
                self.root,
                filename=self.filenames[subdataset],
                md5=self.md5s[subdataset],
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        for subdataset in self.subdatasets:
            extract_archive(os.path.join(self.root, self.filenames[subdataset]))

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.4
        """
        image = sample.get('image')
        if image is not None:
            image = np.rollaxis(sample['image'].numpy(), 0, 3)
        mask = sample.get('mask')
        if mask is not None:
            mask = mask.numpy()
            if mask.ndim == 3:
                mask = np.rollaxis(mask, 0, 3)
            else:
                mask = np.expand_dims(mask, 2)

        num_panels = len(self.layers)
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy()
            num_panels += 1

        fig, axs = plt.subplots(
            1, num_panels, figsize=(num_panels * 4, 5), squeeze=False
        )

        i = 0
        for layer in self.layers:
            if layer == 'naip-new' or layer == 'naip-old':
                if image is None:
                    continue
                img = image[:, :, :3] / 255
                image = image[:, :, 4:]
                axs[0, i].axis('off')
                axs[0, i].imshow(img)
            elif layer == 'landsat-leaf-on' or layer == 'landsat-leaf-off':
                if image is None:
                    continue
                img = image[:, :, [3, 2, 1]] / 3000
                image = image[:, :, 9:]
                axs[0, i].axis('off')
                axs[0, i].imshow(img)
            elif layer == 'nlcd':
                if mask is None:
                    continue
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[0, i].imshow(
                    img, vmin=0, vmax=255, cmap=NLCD.cmap, interpolation='none'
                )
                axs[0, i].axis('off')
            elif layer == 'lc':
                if mask is None:
                    continue
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[0, i].imshow(
                    img, vmin=0, vmax=15, cmap=self.lc_cmap, interpolation='none'
                )
                axs[0, i].axis('off')
            elif layer == 'buildings':
                if mask is None:
                    continue
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[0, i].imshow(img, vmin=0, vmax=1, cmap='gray', interpolation='none')
                axs[0, i].axis('off')
            elif layer == 'prior_from_cooccurrences_101_31_no_osm_no_buildings':
                if mask is None:
                    continue
                img = (mask[:, :, :4] @ self.prior_color_matrix) / 255
                mask = mask[:, :, 4:]
                axs[0, i].imshow(img)
                axs[0, i].axis('off')

            if show_titles:
                if layer == 'prior_from_cooccurrences_101_31_no_osm_no_buildings':
                    axs[0, i].set_title('prior')
                else:
                    axs[0, i].set_title(layer)
            i += 1

        if showing_predictions:
            axs[0, i].imshow(
                predictions, vmin=0, vmax=15, cmap=self.lc_cmap, interpolation='none'
            )
            axs[0, i].axis('off')
            if show_titles:
                axs[0, i].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
