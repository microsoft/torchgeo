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
    <https://www.globalmangrovewatch.org>`_ Version 3.0 dataset provides maps of global
    mangrove extent for eleven epochs between 1996 and 2020, derived from
    L-band Synthetic Aperture Radar (SAR) and optical (Landsat) satellite data.

    Dataset features:

    * Binary mangrove extent maps at 25 m resolution (1/4500 degrees)
    * 11 epochs: 1996, 2007-2010, 2015-2020
    * Global coverage in WGS84 geographic coordinates (EPSG:4326)
    * Tiled as 1° x 1° GeoTIFF files of 4,500 x 4,500 pixels

    Dataset format:

    * single-channel uint8 GeoTIFFs, one ZIP archive per epoch
    * 0 is used as the nodata value

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

    md5s: ClassVar[dict[int, str]] = {
        1996: '7bc81d3aa514d3db5da61e20a36670ca',
        2007: '4d14ca9a5ce2ae605623a1fdcf01d2ef',
        2008: '2643d1b88618da62742dd3ba75fbe4ae',
        2009: '650d969fed2bf85f794db97ff92237a5',
        2010: '99aa8a3c627cd580495ddc2e776385ed',
        2015: 'bf5c888e78db0ce62c86d0bfd2e56b4f',
        2016: '0bc2c9eacf36d9b67aa3a50265a7b74e',
        2017: '9aea4660e3a40e5c3a9ff21de4d8bba6',
        2018: '6bda7daf2b7a3f31451c2d2713d3318b',
        2019: 'd3017880056c6045305b0631058c8cc7',
        2020: 'c85f7528de7df83e5701f3b162ac37b4',
    }

    # Non-mangrove is nodata in the source files, mangrove uses their palette color
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
            checksum: if True, check the MD5 of downloaded and existing archives
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
    def _root(self) -> Path:
        """The single root directory used to download and extract the dataset.

        Returns:
            The root directory.

        Raises:
            AssertionError: If *paths* is not a single root directory.
        """
        assert isinstance(self.paths, str | os.PathLike), (
            'paths must be a single root directory to download or extract data'
        )
        return cast(Path, self.paths)

    @property
    def files(self) -> list[str]:
        """A list of all files in the dataset, restricted to :attr:`years`.

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

    def _zipfile(self, year: int) -> Path:
        """Path of the archive of a single year of the dataset.

        Args:
            year: the year of the archive

        Returns:
            The path of the archive.
        """
        return os.path.join(self._root, self.zipfile_glob.replace('*', str(year)))

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

        # Check if the zip files of the remaining years have already been downloaded
        missing = []
        for year in todo:
            filepath = self._zipfile(year)
            if os.path.exists(filepath):
                if self.checksum and not check_integrity(filepath, self.md5s[year]):
                    raise RuntimeError('Dataset found, but corrupted.')
                self._extract(year)
            else:
                missing.append(year)

        if not missing:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the missing years
        for year in missing:
            self._download(year)
            self._extract(year)

    def _download(self, year: int) -> None:
        """Download a single year of the dataset.

        Args:
            year: the year to download
        """
        download_url(
            self.url.format(year),
            self._root,
            md5=self.md5s[year] if self.checksum else None,
        )

    def _extract(self, year: int) -> None:
        """Extract a single year of the dataset.

        Args:
            year: the year to extract
        """
        extract_archive(self._zipfile(year), self._root)

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
