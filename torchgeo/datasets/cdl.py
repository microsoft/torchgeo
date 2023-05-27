# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import glob
import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive


class CDL(RasterDataset):
    """Cropland Data Layer (CDL) dataset.

    The `Cropland Data Layer
    <https://data.nal.usda.gov/dataset/cropscape-cropland-data-layer>`__, hosted on
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

    * https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#Section1_14.0
    """  # noqa: E501

    filename_glob = "*_30m_cdls.tif"
    filename_regex = r"""
        ^(?P<date>\d+)
        _30m_cdls\..*$
    """
    zipfile_glob = "*_30m_cdls.zip"
    date_format = "%Y"
    is_image = False

    url = "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{}_30m_cdls.zip"  # noqa: E501
    md5s = {
        2022: "754cf50670cdfee511937554785de3e6",
        2021: "27606eab08fe975aa138baad3e5dfcd8",
        2020: "483ee48c503aa81b684225179b402d42",
        2019: "a5168a2fc93acbeaa93e24eee3d8c696",
        2018: "4ad0d7802a9bb751685eb239b0fa8609",
        2017: "d173f942a70f94622f9b8290e7548684",
        2016: "fddc5dff0bccc617d70a12864c993e51",
        2015: "2e92038ab62ba75e1687f60eecbdd055",
        2014: "50bdf9da84ebd0457ddd9e0bf9bbcc1f",
        2013: "7be66c650416dc7c4a945dd7fd93c5b7",
        2012: "286504ff0512e9fe1a1975c635a1bec2",
        2011: "517bad1a99beec45d90abb651fb1f0e3",
        2010: "98d354c5a62c9e3e40ccadce265c721c",
        2009: "663c8a5fdd92ebfc0d6bee008586d19a",
        2008: "0610f2f17ab60a9fbb3baeb7543993a4",
    }

    cmap = [
        (0, 0, 0, 0),
        (255, 211, 0, 255),
        (255, 38, 38, 255),
        (0, 168, 228, 255),
        (255, 158, 11, 255),
        (38, 112, 0, 255),
        (255, 255, 0, 255),
        (112, 165, 0, 255),
        (0, 175, 75, 255),
        (221, 165, 11, 255),
        (221, 165, 11, 255),
        (126, 211, 255, 255),
        (226, 0, 124, 255),
        (137, 98, 84, 255),
        (216, 181, 107, 255),
        (165, 112, 0, 255),
        (214, 158, 188, 255),
        (112, 112, 0, 255),
        (172, 0, 124, 255),
        (160, 89, 137, 255),
        (112, 0, 73, 255),
        (214, 158, 188, 255),
        (209, 255, 0, 255),
        (126, 153, 255, 255),
        (214, 214, 0, 255),
        (209, 255, 0, 255),
        (0, 175, 75, 255),
        (255, 165, 226, 255),
        (165, 242, 140, 255),
        (0, 175, 75, 255),
        (214, 158, 188, 255),
        (168, 0, 228, 255),
        (165, 0, 0, 255),
        (112, 38, 0, 255),
        (0, 175, 75, 255),
        (177, 126, 255, 255),
        (112, 38, 0, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 204, 102, 255),
        (255, 102, 102, 255),
        (0, 175, 75, 255),
        (0, 221, 175, 255),
        (84, 255, 0, 255),
        (242, 163, 119, 255),
        (255, 102, 102, 255),
        (0, 175, 75, 255),
        (126, 211, 255, 255),
        (232, 191, 255, 255),
        (175, 255, 221, 255),
        (0, 175, 75, 255),
        (191, 191, 119, 255),
        (147, 204, 147, 255),
        (198, 214, 158, 255),
        (204, 191, 163, 255),
        (255, 0, 255, 255),
        (255, 142, 170, 255),
        (186, 0, 79, 255),
        (112, 68, 137, 255),
        (0, 119, 119, 255),
        (177, 154, 112, 255),
        (255, 255, 126, 255),
        (181, 112, 91, 255),
        (0, 165, 130, 255),
        (233, 214, 175, 255),
        (177, 154, 112, 255),
        (242, 242, 242, 255),
        (154, 154, 154, 255),
        (75, 112, 163, 255),
        (126, 177, 177, 255),
        (232, 255, 191, 255),
        (0, 255, 255, 255),
        (75, 112, 163, 255),
        (211, 226, 249, 255),
        (154, 154, 154, 255),
        (154, 154, 154, 255),
        (154, 154, 154, 255),
        (154, 154, 154, 255),
        (204, 191, 163, 255),
        (147, 204, 147, 255),
        (147, 204, 147, 255),
        (147, 204, 147, 255),
        (198, 214, 158, 255),
        (232, 255, 191, 255),
        (126, 177, 177, 255),
        (126, 177, 177, 255),
        (0, 255, 140, 255),
        (214, 158, 188, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 142, 170, 255),
        (51, 73, 51, 255),
        (228, 112, 38, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (102, 153, 76, 255),
        (255, 102, 102, 255),
        (177, 154, 112, 255),
        (255, 142, 170, 255),
        (255, 102, 102, 255),
        (255, 142, 170, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 142, 170, 255),
        (0, 175, 75, 255),
        (255, 211, 0, 255),
        (255, 211, 0, 255),
        (255, 102, 102, 255),
        (255, 210, 0, 255),
        (255, 102, 102, 255),
        (137, 98, 84, 255),
        (255, 102, 102, 255),
        (255, 38, 38, 255),
        (226, 0, 124, 255),
        (255, 158, 11, 255),
        (255, 158, 11, 255),
        (165, 112, 0, 255),
        (255, 211, 0, 255),
        (165, 112, 0, 255),
        (38, 112, 0, 255),
        (38, 112, 0, 255),
        (255, 211, 0, 255),
        (0, 0, 153, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (255, 102, 102, 255),
        (38, 112, 0, 255),
    ]

    ordinal_label_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        10: 7,
        11: 8,
        12: 9,
        13: 10,
        14: 11,
        21: 12,
        22: 13,
        23: 14,
        24: 15,
        25: 16,
        26: 17,
        27: 18,
        28: 19,
        29: 20,
        30: 21,
        31: 22,
        32: 23,
        33: 24,
        34: 25,
        35: 26,
        36: 27,
        37: 28,
        38: 29,
        39: 30,
        41: 31,
        42: 32,
        43: 33,
        44: 34,
        45: 35,
        46: 36,
        47: 37,
        48: 38,
        49: 39,
        50: 40,
        51: 41,
        52: 42,
        53: 43,
        54: 44,
        55: 45,
        56: 46,
        57: 47,
        58: 48,
        59: 49,
        60: 50,
        61: 51,
        63: 52,
        64: 53,
        65: 54,
        66: 55,
        67: 56,
        68: 57,
        69: 58,
        70: 59,
        71: 60,
        72: 61,
        74: 62,
        75: 63,
        76: 64,
        77: 65,
        81: 66,
        82: 67,
        83: 68,
        87: 69,
        88: 70,
        92: 71,
        111: 72,
        112: 73,
        121: 74,
        122: 75,
        123: 76,
        124: 77,
        131: 78,
        141: 79,
        142: 80,
        143: 81,
        152: 82,
        176: 83,
        190: 84,
        195: 85,
        204: 86,
        205: 87,
        206: 88,
        207: 89,
        208: 90,
        209: 91,
        210: 92,
        211: 93,
        212: 94,
        213: 95,
        214: 96,
        215: 97,
        216: 98,
        217: 99,
        218: 100,
        219: 101,
        220: 102,
        221: 103,
        222: 104,
        223: 105,
        224: 106,
        225: 107,
        226: 108,
        227: 109,
        228: 110,
        229: 111,
        230: 112,
        231: 113,
        232: 114,
        233: 115,
        234: 116,
        235: 117,
        236: 118,
        237: 119,
        238: 120,
        239: 121,
        240: 122,
        241: 123,
        242: 124,
        243: 125,
        244: 126,
        245: 127,
        246: 128,
        247: 129,
        248: 130,
        249: 131,
        250: 132,
        254: 133,
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        years: list[int] = [2022],
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            years: list of years for which to use cdl layer
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails

        .. versionadded:: 0.5
           The *years* parameter.
        """
        assert set(years).issubset(self.md5s.keys()), (
            "CDL data product only exists for the following years: "
            f"{list(self.md5s.keys())}."
        )
        self.years = years
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

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

        mask = sample["mask"]
        for k, v in self.ordinal_label_map.items():
            mask[mask == k] = v

        sample["mask"] = mask

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        exists = []
        for year in self.years:
            filename_year = self.filename_glob.replace("*", str(year))
            pathname = os.path.join(self.root, "**", filename_year)
            for fname in glob.iglob(pathname, recursive=True):
                if not fname.endswith(".zip"):
                    exists.append(True)

        if len(exists) == len(self.years):
            return

        # Check if the zip files have already been downloaded
        exists = []
        for year in self.years:
            pathname = os.path.join(
                self.root, self.zipfile_glob.replace("*", str(year))
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
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for year in self.years:
            download_url(
                self.url.format(year),
                self.root,
                md5=self.md5s[year] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        for year in self.years:
            zipfile_name = self.zipfile_glob.replace("*", str(year))
            pathname = os.path.join(self.root, zipfile_name)
            extract_archive(pathname, self.root)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
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
        mask = sample["mask"].squeeze().numpy()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze().numpy()
            ncols = 2

        kwargs = {
            "cmap": ListedColormap(np.array(self.cmap) / 255),
            "vmin": 0,
            "vmax": len(self.cmap) - 1,
            "interpolation": "none",
        }

        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 4, 4), squeeze=False
        )

        axs[0, 0].imshow(mask, **kwargs)
        axs[0, 0].axis("off")

        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(pred, **kwargs)
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
