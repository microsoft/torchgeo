# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Global Mangrove Watch dataset."""

import os
import re
from collections.abc import Callable, Iterable, Sequence
from typing import ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import Path, Sample, check_integrity, download_url, extract_archive


class GlobalMangroveWatch(RasterDataset):
    """Global Mangrove Watch (GMW) dataset.

    The `Global Mangrove Watch
    <https://www.globalmangrovewatch.org>`__ Version 3.0 dataset provides maps of global
    mangrove extent for eleven epochs between 1996 and 2020, derived from
    L-band Synthetic Aperture Radar (SAR) and optical (Landsat) satellite data.

    Dataset features:

    * Binary mangrove extent maps at 25 m resolution (1/4500 degrees)
    * 11 epochs: 1996, 2007-2010, 2015-2020
    * Global coverage in WGS84 geographic coordinates (EPSG:4326)
    * Tiled as 1° x 1° GeoTIFF files of 4,500 x 4,500 pixels

    Dataset format:

    * single-channel uint8 GeoTIFFs, one ZIP archive per epoch

    Dataset classes:

    0. Non-mangrove
    1. Mangrove

    If you use this dataset in your research, please cite it using the
    following format:

    * https://doi.org/10.5281/zenodo.6894273

    .. versionadded:: 0.10
    """

    filename_glob = 'GMW_*_*_v3.tif'
    filename_regex = r'GMW_[NS]\d+[EW]\d+_(?P<date>\d{4})_v3\.tif'
    zipfile_glob = 'gmw_v3_*_gtiff.zip'
    date_format = '%Y'
    is_image = False

    url = 'https://zenodo.org/records/6894273/files/gmw_v3_{}_gtiff.zip'

    all_years: ClassVar[tuple[int, ...]] = (
        1996,
        2007,
        2008,
        2009,
        2010,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
    )

    sha256s: ClassVar[dict[int, str]] = {
        1996: '00923d94d3861bb847faf0a91311d39d49242e824f63672e4dbec514df9e46c9',
        2007: 'f150c316f682de58cb58e7d32834472d889a0ff709b66a5b9a82b9bea8a05d9b',
        2008: 'bf73e12e50c085a1bf6093d99b5b76abf85fc6c0c5925ac302bc555db114d635',
        2009: '8850e43e2455d73cea10a09d1bf14445c83f8eb0c4c89b1dfacd8030c7470c15',
        2010: '5da7f9d1bfd28aa2002db36153ef7407abb4926fb96474eb3647b4df070f1e15',
        2015: '7c8945377261c5ef37fd5a3e3a791d4c48ad0b10c7e81feabc605b569ae08166',
        2016: 'c02e04a88cbf83bc74ee7e99e42c7902b27183505d525d93c5170d2049ae5a7f',
        2017: 'bf9fb170d147b0fa9b6d2e67c644c09427394207416ab240932bcca6b426548b',
        2018: 'b65fe68ce921c2e4e3ef199b3ec9e840231ef33cdecf768252eaa33286bbe720',
        2019: '5a6e31dcd808e5efd19eb85cbc97275b70a5861847d28cdc076030aae33df102',
        2020: '97d783a0904fc97738d50bcb3b0a3e2ec0bfc87a2d5d09e349d59a330619f66b',
    }

    # Mangrove uses the color from the official GMW palette
    cmap = ListedColormap(np.array([(255, 255, 255, 255), (0, 150, 0, 255)]) / 255)

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        years: Sequence[int] = (2020,),
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
        time_series: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format.
                If a single float is provided, it is used for both x and y resolution.
                (defaults to the resolution of the first file found)
            years: years for which to use GMW layers
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the SHA256 of downloaded and existing archives
                (may be slow)
            time_series: if True, stack data along the time series dimension
                [T, C, H, W]. If False, merge data into a [C, H, W] mosaic.

        Raises:
            AssertionError: If ``years`` are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
            RuntimeError: If an existing archive is corrupted.
        """
        assert set(years) <= set(self.all_years), (
            f'GMW data product only exists for the following years: '
            f'{list(self.all_years)}.'
        )

        self.paths = paths
        self.years = tuple(years)
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    @property
    def files(self) -> list[str]:
        """A list of all files in the dataset, restricted to *years*.

        Returns:
            All files belonging to one of the requested years.
        """
        return [f for f in super().files if self._year(f) in self.years]

    def _year(self, filepath: Path) -> int | None:
        """Extract the year of a file from its filename.

        Args:
            filepath: path of the file

        Returns:
            The year of the file, or None if the filename does not match
            :attr:`filename_regex`.
        """
        match = re.match(self.filename_regex, os.path.basename(filepath), re.VERBOSE)
        return int(match.group('date')) if match else None

    def _verify(self) -> None:
        """Verify dataset integrity.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            RuntimeError: If an existing archive is corrupted.
        """
        # Check which years have already been extracted
        extracted = {self._year(f) for f in self.files}
        todo = [year for year in self.years if year not in extracted]
        if not todo:
            return

        assert isinstance(self.paths, str | os.PathLike), (
            'paths must be a single root directory to download or extract data'
        )
        paths = cast(Path, self.paths)

        for year in todo:
            filepath = os.path.join(paths, self.zipfile_glob.replace('*', str(year)))

            # Check if the zip file has already been downloaded
            if os.path.exists(filepath):
                if self.checksum and not check_integrity(
                    filepath, sha256=self.sha256s[year]
                ):
                    raise RuntimeError('Dataset found, but corrupted.')
            elif self.download:
                download_url(
                    self.url.format(year),
                    paths,
                    sha256=self.sha256s[year] if self.checksum else None,
                )
            else:
                raise DatasetNotFoundError(self)

            extract_archive(filepath, paths)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by the dataset
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # Masks are [H, W] for a mosaic and [T, H, W] for a time series,
        # both are reshaped to a [T, H, W] stack of frames to plot.
        panels = [('Mask', sample['mask'].reshape(-1, *sample['mask'].shape[-2:]))]
        if 'prediction' in sample:
            pred = sample['prediction']
            panels.append(('Prediction', pred.reshape(-1, *pred.shape[-2:])))

        nrows = len(panels)
        ncols = panels[0][1].shape[0]

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), squeeze=False
        )

        kwargs = {'cmap': self.cmap, 'vmin': 0, 'vmax': 1, 'interpolation': 'none'}

        for row, (title, frames) in enumerate(panels):
            for col in range(ncols):
                axs[row, col].imshow(frames[col], **kwargs)
                axs[row, col].axis('off')
                if show_titles:
                    axs[row, col].set_title(title if ncols == 1 else f'{title} {col}')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
