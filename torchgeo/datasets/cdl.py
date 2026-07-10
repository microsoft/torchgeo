# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import os
from collections.abc import Callable, Iterable
from typing import ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
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

    cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 255),
                (255, 210, 0, 255),
                (255, 36, 36, 255),
                (0, 168, 226, 255),
                (255, 158, 9, 255),
                (36, 110, 0, 255),
                (255, 255, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (110, 163, 0, 255),
                (0, 173, 73, 255),
                (221, 163, 9, 255),
                (221, 163, 9, 255),
                (124, 210, 255, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (224, 0, 122, 255),
                (135, 96, 82, 255),
                (214, 181, 107, 255),
                (163, 110, 0, 255),
                (212, 158, 186, 255),
                (110, 110, 0, 255),
                (172, 0, 122, 255),
                (159, 87, 135, 255),
                (110, 0, 71, 255),
                (212, 158, 186, 255),
                (209, 255, 0, 255),
                (124, 153, 255, 255),
                (212, 212, 0, 255),
                (209, 255, 0, 255),
                (0, 173, 73, 255),
                (255, 163, 224, 255),
                (163, 240, 138, 255),
                (0, 173, 73, 255),
                (212, 158, 186, 255),
                (0, 0, 0, 255),
                (168, 0, 226, 255),
                (163, 0, 0, 255),
                (110, 36, 0, 255),
                (0, 173, 73, 255),
                (175, 124, 255, 255),
                (110, 36, 0, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 204, 102, 255),
                (255, 102, 102, 255),
                (0, 173, 73, 255),
                (0, 221, 173, 255),
                (82, 255, 0, 255),
                (240, 161, 119, 255),
                (255, 102, 102, 255),
                (0, 173, 73, 255),
                (124, 210, 255, 255),
                (232, 189, 255, 255),
                (173, 255, 221, 255),
                (0, 173, 73, 255),
                (189, 189, 119, 255),
                (0, 0, 0, 255),
                (146, 204, 146, 255),
                (197, 212, 158, 255),
                (204, 189, 161, 255),
                (255, 0, 255, 255),
                (255, 142, 170, 255),
                (184, 0, 79, 255),
                (110, 68, 135, 255),
                (0, 119, 119, 255),
                (175, 154, 110, 255),
                (255, 255, 124, 255),
                (0, 0, 0, 255),
                (181, 110, 91, 255),
                (0, 163, 130, 255),
                (233, 212, 173, 255),
                (175, 154, 110, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (240, 240, 240, 255),
                (154, 154, 154, 255),
                (73, 110, 161, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (124, 175, 175, 255),
                (232, 255, 189, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 255, 255, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (73, 110, 161, 255),
                (210, 224, 248, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (154, 154, 154, 255),
                (154, 154, 154, 255),
                (154, 154, 154, 255),
                (154, 154, 154, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (204, 189, 161, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (146, 204, 146, 255),
                (146, 204, 146, 255),
                (146, 204, 146, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (197, 212, 158, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (232, 255, 189, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (124, 175, 175, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (124, 175, 175, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 255, 138, 255),
                (212, 158, 186, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 142, 170, 255),
                (51, 71, 51, 255),
                (226, 110, 36, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (102, 153, 74, 255),
                (255, 102, 102, 255),
                (175, 154, 110, 255),
                (255, 142, 170, 255),
                (255, 102, 102, 255),
                (255, 142, 170, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 142, 170, 255),
                (0, 173, 73, 255),
                (255, 210, 0, 255),
                (255, 210, 0, 255),
                (255, 102, 102, 255),
                (255, 210, 0, 255),
                (255, 102, 102, 255),
                (135, 96, 82, 255),
                (255, 102, 102, 255),
                (255, 36, 36, 255),
                (224, 0, 122, 255),
                (255, 158, 9, 255),
                (255, 158, 9, 255),
                (163, 110, 0, 255),
                (255, 210, 0, 255),
                (163, 110, 0, 255),
                (36, 110, 0, 255),
                (36, 110, 0, 255),
                (255, 210, 0, 255),
                (0, 0, 153, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (255, 102, 102, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (36, 110, 0, 255),
                (0, 0, 0, 255),
            ]
        )
        / 255
    )

    valid_classes = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        74,
        75,
        76,
        77,
        81,
        82,
        83,
        87,
        88,
        92,
        111,
        112,
        121,
        122,
        123,
        124,
        131,
        141,
        142,
        143,
        152,
        176,
        190,
        195,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        254,
    )

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        years: list[int] = [2023],
        classes: list[int] = list(valid_classes),
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
        assert set(classes) <= set(self.valid_classes), (
            f'Only the following classes are valid: {self.valid_classes}.'
        )
        assert 0 in classes, 'Classes must include the background class: 0'

        self.paths = paths
        self.years = years
        self.classes = classes
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.zeros(self.valid_classes[-1] + 1, dtype=self.dtype)
        self.inverse_map = torch.zeros(len(classes), dtype=self.dtype)

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.inverse_map[v] = k

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
        mask = self.inverse_map[sample['mask']]
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = self.inverse_map[sample['prediction']]
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 4, 4), squeeze=False)
        kwargs = {'cmap': self.cmap, 'vmin': 0, 'vmax': 255, 'interpolation': 'none'}

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
