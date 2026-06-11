# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Global Mangrove Watch dataset."""

import os
from collections.abc import Callable, Iterable
from typing import ClassVar, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import Path, Sample, download_url, extract_archive


class GlobalMangroveWatch(RasterDataset):
    """Global Mangrove Watch (GMW) dataset.

    The `Global Mangrove Watch
    <https://www.globalmangrovewatch.org>`_ dataset provides maps of global
    mangrove extent for eleven epochs between 1996 and 2020, derived from
    L-band Synthetic Aperture Radar (SAR) and optical (Landsat) satellite data.

    Dataset features:

    * Binary mangrove extent maps at 25m resolution
    * 11 epochs: 1996, 2007-2010, 2015-2020
    * Global coverage in WGS84 geographic coordinates
    * Tiled as 1° x 1° GeoTIFF files

    Dataset classes:

    0. Non-mangrove
    #. Mangrove

    If you use this dataset in your research, please cite it using the
    following format:

    * https://doi.org/10.5281/zenodo.6894273

    .. versionadded:: 0.9
    """

    filename_glob = 'gmw_v3_*_*.tif'
    filename_regex = r'gmw_v3_(?P<date>\d{4})_[NS]\d+[EW]\d+\.tif'
    zipfile_glob = 'gmw_v3_*_gtiff.zip'
    date_format = '%Y'
    is_image = False

    url = 'https://zenodo.org/records/6894273/files/gmw_v3_{}_gtiff.zip?download=1'

    all_years: ClassVar[list[int]] = [
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
    ]

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

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (255, 255, 255, 255),
        1: (0, 128, 0, 255),
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        years: list[int] = [2020],
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
            years: list of years for which to use GMW layers
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
            time_series: if True, stack data along the time series dimension
                [T, C, H, W]. If False, merge data into a [C, H, W] mosaic.

        Raises:
            AssertionError: if ``years`` are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(years) <= set(self.all_years), (
            f'GMW data product only exists for the following years: {self.all_years}.'
        )

        self.paths = paths
        self.years = years
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def _verify(self) -> None:
        """Verify dataset integrity."""
        if self.files:
            return

        exists = []
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        for year in self.years:
            pathname = os.path.join(paths, self.zipfile_glob.replace('*', str(year)))
            if os.path.exists(pathname):
                exists.append(True)
                self._extract()
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        for year in self.years:
            download_url(
                self.url.format(year),
                paths,
                md5=self.md5s[year] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        paths = cast(Path, self.paths)
        for year in self.years:
            zipfile_name = self.zipfile_glob.replace('*', str(year))
            pathname = os.path.join(paths, zipfile_name)
            extract_archive(pathname, paths)

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
        mask = sample['mask'].squeeze()
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction'].squeeze()
            ncols = 2

        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 4, 4), squeeze=False
        )

        cmap_colors = torch.tensor(
            [self.cmap[i] for i in range(len(self.cmap))], dtype=torch.uint8
        )

        axs[0, 0].imshow(cmap_colors[mask], interpolation='none')
        axs[0, 0].axis('off')
        if show_titles:
            axs[0, 0].set_title('Mask')

        if showing_predictions:
            axs[0, 1].imshow(cmap_colors[pred], interpolation='none')
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
