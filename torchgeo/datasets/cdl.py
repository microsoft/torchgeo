# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import os
from collections.abc import Callable, Iterable
from typing import ClassVar, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import GeoSlice, Path, Sample, download_url, extract_archive


class CDL(RasterDataset):
    """Cropland Data Layer (CDL) dataset.

    The `Cropland Data Layer
    <https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php>`__, hosted on
    `CropScape <https://nassgeodata.gmu.edu/CropScape/%3B>`_, provides a raster,
    geo-referenced, crop-specific land cover map for the continental United States. The
    CDL also includes a crop mask layer and planting frequency layers, as well as
    boundary, water and road layers. The Boundary Layer options provided are County,
    Agricultural Statistics Districts (ASD), State, and Region. The data is created
    annually using moderate resolution satellite imagery and extensive agricultural
    ground truth.

    The dataset contains 134 classes, for a description of the classes see the
    xls file at the top of
    `this page <https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php>`_.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.1
    """

    filename_glob = '*_30m_cdls.tif'
    filename_regex = r"""
        ^(?P<date>\d+)
        _30m_cdls\..*$
    """
    zipfile_glob = '*_30m_cdls.zip'
    date_format = '%Y'
    is_image = False

    url = 'https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{}_30m_cdls.zip'
    md5s: ClassVar[dict[int, str]] = {
        2024: '841cd0cb8d4a9129cb1e4cfa0e40c286',
        2023: '8c7685d6278d50c554f934b16a6076b7',
        2022: '754cf50670cdfee511937554785de3e6',
        2021: '27606eab08fe975aa138baad3e5dfcd8',
        2020: '483ee48c503aa81b684225179b402d42',
        2019: 'a5168a2fc93acbeaa93e24eee3d8c696',
        2018: '4ad0d7802a9bb751685eb239b0fa8609',
        2017: 'd173f942a70f94622f9b8290e7548684',
        2016: 'fddc5dff0bccc617d70a12864c993e51',
        2015: '2e92038ab62ba75e1687f60eecbdd055',
        2014: '50bdf9da84ebd0457ddd9e0bf9bbcc1f',
        2013: '7be66c650416dc7c4a945dd7fd93c5b7',
        2012: '286504ff0512e9fe1a1975c635a1bec2',
        2011: '517bad1a99beec45d90abb651fb1f0e3',
        2010: '98d354c5a62c9e3e40ccadce265c721c',
        2009: '663c8a5fdd92ebfc0d6bee008586d19a',
        2008: '0610f2f17ab60a9fbb3baeb7543993a4',
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        years: list[int] = [2023],
        classes: list[int] = list(range(256)),
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
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            years: list of years for which to use cdl layer
            classes: list of classes to include, the rest will be mapped to 0
                (defaults to all classes)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            time_series: if True, stack data along the time series dimension
                [T, C, H, W]. If False, merge data into a [C, H, W] mosaic.

        Raises:
            AssertionError: if ``years`` or ``classes`` are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.9
           The *time_series* parameter.

        .. versionadded:: 0.5
           The *years* and *classes* parameters.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        assert set(years) <= self.md5s.keys(), (
            'CDL data product only exists for the following years: '
            f'{list(self.md5s.keys())}.'
        )
        assert set(classes) <= set(range(256)), (
            f'Only the following classes are valid: {list(range(256))}.'
        )
        assert 0 in classes, 'Classes must include the background class: 0'

        self.paths = paths
        self.years = years
        self.classes = classes
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.zeros(256, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        sample = super().__getitem__(index)
        sample['mask'] = self.ordinal_map[sample['mask']]
        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if self.files:
            return

        # Check if the zip files have already been downloaded
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
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
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

        axs[0, 0].imshow(self.ordinal_cmap[mask], interpolation='none')
        axs[0, 0].axis('off')

        if show_titles:
            axs[0, 0].set_title('Mask')

        if showing_predictions:
            axs[0, 1].imshow(self.ordinal_cmap[pred], interpolation='none')
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
