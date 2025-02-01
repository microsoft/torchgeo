# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import os
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import BoundingBox, Path, download_url, extract_archive


class CDL(RasterDataset):
    """Cropland Data Layer (CDL) dataset.

    The `Cropland Data Layer
    <https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php>`__, hosted on
    `CropScape <https://nassgeodata.gmu.edu/CropScape/>`_, provides a raster,
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

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (0, 0, 0, 255),
        1: (255, 211, 0, 255),
        2: (255, 37, 37, 255),
        3: (0, 168, 226, 255),
        4: (255, 158, 9, 255),
        5: (37, 111, 0, 255),
        6: (255, 255, 0, 255),
        10: (111, 166, 0, 255),
        11: (0, 175, 73, 255),
        12: (222, 166, 9, 255),
        13: (222, 166, 9, 255),
        14: (124, 211, 255, 255),
        21: (226, 0, 124, 255),
        22: (137, 96, 83, 255),
        23: (217, 181, 107, 255),
        24: (166, 111, 0, 255),
        25: (213, 158, 188, 255),
        26: (111, 111, 0, 255),
        27: (171, 0, 124, 255),
        28: (160, 88, 137, 255),
        29: (111, 0, 73, 255),
        30: (213, 158, 188, 255),
        31: (209, 255, 0, 255),
        32: (124, 153, 255, 255),
        33: (213, 213, 0, 255),
        34: (209, 255, 0, 255),
        35: (0, 175, 73, 255),
        36: (255, 166, 226, 255),
        37: (166, 241, 139, 255),
        38: (0, 175, 73, 255),
        39: (213, 158, 188, 255),
        41: (168, 0, 226, 255),
        42: (166, 0, 0, 255),
        43: (111, 37, 0, 255),
        44: (0, 175, 73, 255),
        45: (175, 124, 255, 255),
        46: (111, 37, 0, 255),
        47: (255, 102, 102, 255),
        48: (255, 102, 102, 255),
        49: (255, 204, 102, 255),
        50: (255, 102, 102, 255),
        51: (0, 175, 73, 255),
        52: (0, 222, 175, 255),
        53: (83, 255, 0, 255),
        54: (241, 162, 120, 255),
        55: (255, 102, 102, 255),
        56: (0, 175, 73, 255),
        57: (124, 211, 255, 255),
        58: (232, 190, 255, 255),
        59: (175, 255, 222, 255),
        60: (0, 175, 73, 255),
        61: (190, 190, 120, 255),
        63: (147, 204, 147, 255),
        64: (198, 213, 158, 255),
        65: (204, 190, 162, 255),
        66: (255, 0, 255, 255),
        67: (255, 143, 171, 255),
        68: (185, 0, 79, 255),
        69: (111, 69, 137, 255),
        70: (0, 120, 120, 255),
        71: (175, 153, 111, 255),
        72: (255, 255, 124, 255),
        74: (181, 111, 92, 255),
        75: (0, 166, 130, 255),
        76: (232, 213, 175, 255),
        77: (175, 153, 111, 255),
        81: (241, 241, 241, 255),
        82: (153, 153, 153, 255),
        83: (73, 111, 162, 255),
        87: (124, 175, 175, 255),
        88: (232, 255, 190, 255),
        92: (0, 255, 255, 255),
        111: (73, 111, 162, 255),
        112: (211, 226, 249, 255),
        121: (153, 153, 153, 255),
        122: (153, 153, 153, 255),
        123: (153, 153, 153, 255),
        124: (153, 153, 153, 255),
        131: (204, 190, 162, 255),
        141: (147, 204, 147, 255),
        142: (147, 204, 147, 255),
        143: (147, 204, 147, 255),
        152: (198, 213, 158, 255),
        176: (232, 255, 190, 255),
        190: (124, 175, 175, 255),
        195: (124, 175, 175, 255),
        204: (0, 255, 139, 255),
        205: (213, 158, 188, 255),
        206: (255, 102, 102, 255),
        207: (255, 102, 102, 255),
        208: (255, 102, 102, 255),
        209: (255, 102, 102, 255),
        210: (255, 143, 171, 255),
        211: (51, 73, 51, 255),
        212: (226, 111, 37, 255),
        213: (255, 102, 102, 255),
        214: (255, 102, 102, 255),
        215: (102, 153, 77, 255),
        216: (255, 102, 102, 255),
        217: (175, 153, 111, 255),
        218: (255, 143, 171, 255),
        219: (255, 102, 102, 255),
        220: (255, 143, 171, 255),
        221: (255, 102, 102, 255),
        222: (255, 102, 102, 255),
        223: (255, 143, 171, 255),
        224: (0, 175, 73, 255),
        225: (255, 211, 0, 255),
        226: (255, 211, 0, 255),
        227: (255, 102, 102, 255),
        228: (255, 211, 0, 255),
        229: (255, 102, 102, 255),
        230: (137, 96, 83, 255),
        231: (255, 102, 102, 255),
        232: (255, 37, 37, 255),
        233: (226, 0, 124, 255),
        234: (255, 158, 9, 255),
        235: (255, 158, 9, 255),
        236: (166, 111, 0, 255),
        237: (255, 211, 0, 255),
        238: (166, 111, 0, 255),
        239: (37, 111, 0, 255),
        240: (37, 111, 0, 255),
        241: (255, 211, 0, 255),
        242: (0, 0, 153, 255),
        243: (255, 102, 102, 255),
        244: (255, 102, 102, 255),
        245: (255, 102, 102, 255),
        246: (255, 102, 102, 255),
        247: (255, 102, 102, 255),
        248: (255, 102, 102, 255),
        249: (255, 102, 102, 255),
        250: (255, 102, 102, 255),
        254: (37, 111, 0, 255),
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        years: list[int] = [2023],
        classes: list[int] = list(cmap.keys()),
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            years: list of years for which to use cdl layer
            classes: list of classes to include, the rest will be mapped to 0
                (defaults to all classes)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``years`` or ``classes`` are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.5
           The *years* and *classes* parameters.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        assert set(years) <= self.md5s.keys(), (
            'CDL data product only exists for the following years: '
            f'{list(self.md5s.keys())}.'
        )
        assert set(classes) <= self.cmap.keys(), (
            f'Only the following classes are valid: {list(self.cmap.keys())}.'
        )
        assert 0 in classes, 'Classes must include the background class: 0'

        self.paths = paths
        self.years = years
        self.classes = classes
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)

        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        sample = super().__getitem__(query)
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
        for year in self.years:
            pathname = os.path.join(
                self.paths, self.zipfile_glob.replace('*', str(year))
            )
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
        for year in self.years:
            download_url(
                self.url.format(year),
                self.paths,
                md5=self.md5s[year] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        for year in self.years:
            zipfile_name = self.zipfile_glob.replace('*', str(year))
            pathname = os.path.join(self.paths, zipfile_name)
            extract_archive(pathname, self.paths)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
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
