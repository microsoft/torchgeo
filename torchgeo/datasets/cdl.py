# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import download_url, extract_archive


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
    md5s = [
        (2021, "27606eab08fe975aa138baad3e5dfcd8"),
        (2020, "483ee48c503aa81b684225179b402d42"),
        (2019, "a5168a2fc93acbeaa93e24eee3d8c696"),
        (2018, "4ad0d7802a9bb751685eb239b0fa8609"),
        (2017, "d173f942a70f94622f9b8290e7548684"),
        (2016, "fddc5dff0bccc617d70a12864c993e51"),
        (2015, "2e92038ab62ba75e1687f60eecbdd055"),
        (2014, "50bdf9da84ebd0457ddd9e0bf9bbcc1f"),
        (2013, "7be66c650416dc7c4a945dd7fd93c5b7"),
        (2012, "286504ff0512e9fe1a1975c635a1bec2"),
        (2011, "517bad1a99beec45d90abb651fb1f0e3"),
        (2010, "98d354c5a62c9e3e40ccadce265c721c"),
        (2009, "663c8a5fdd92ebfc0d6bee008586d19a"),
        (2008, "0610f2f17ab60a9fbb3baeb7543993a4"),
    ]

    cmap = {
        0: (0, 0, 0, 0),
        1: (255, 211, 0, 255),
        2: (255, 38, 38, 255),
        3: (0, 168, 228, 255),
        4: (255, 158, 11, 255),
        5: (38, 112, 0, 255),
        6: (255, 255, 0, 255),
        7: (0, 0, 0, 255),
        8: (0, 0, 0, 255),
        9: (0, 0, 0, 255),
        10: (112, 165, 0, 255),
        11: (0, 175, 75, 255),
        12: (221, 165, 11, 255),
        13: (221, 165, 11, 255),
        14: (126, 211, 255, 255),
        15: (0, 0, 0, 255),
        16: (0, 0, 0, 255),
        17: (0, 0, 0, 255),
        18: (0, 0, 0, 255),
        19: (0, 0, 0, 255),
        20: (0, 0, 0, 255),
        21: (226, 0, 124, 255),
        22: (137, 98, 84, 255),
        23: (216, 181, 107, 255),
        24: (165, 112, 0, 255),
        25: (214, 158, 188, 255),
        26: (112, 112, 0, 255),
        27: (172, 0, 124, 255),
        28: (160, 89, 137, 255),
        29: (112, 0, 73, 255),
        30: (214, 158, 188, 255),
        31: (209, 255, 0, 255),
        32: (126, 153, 255, 255),
        33: (214, 214, 0, 255),
        34: (209, 255, 0, 255),
        35: (0, 175, 75, 255),
        36: (255, 165, 226, 255),
        37: (165, 242, 140, 255),
        38: (0, 175, 75, 255),
        39: (214, 158, 188, 255),
        40: (0, 0, 0, 255),
        41: (168, 0, 228, 255),
        42: (165, 0, 0, 255),
        43: (112, 38, 0, 255),
        44: (0, 175, 75, 255),
        45: (177, 126, 255, 255),
        46: (112, 38, 0, 255),
        47: (255, 102, 102, 255),
        48: (255, 102, 102, 255),
        49: (255, 204, 102, 255),
        50: (255, 102, 102, 255),
        51: (0, 175, 75, 255),
        52: (0, 221, 175, 255),
        53: (84, 255, 0, 255),
        54: (242, 163, 119, 255),
        55: (255, 102, 102, 255),
        56: (0, 175, 75, 255),
        57: (126, 211, 255, 255),
        58: (232, 191, 255, 255),
        59: (175, 255, 221, 255),
        60: (0, 175, 75, 255),
        61: (191, 191, 119, 255),
        62: (0, 0, 0, 255),
        63: (147, 204, 147, 255),
        64: (198, 214, 158, 255),
        65: (204, 191, 163, 255),
        66: (255, 0, 255, 255),
        67: (255, 142, 170, 255),
        68: (186, 0, 79, 255),
        69: (112, 68, 137, 255),
        70: (0, 119, 119, 255),
        71: (177, 154, 112, 255),
        72: (255, 255, 126, 255),
        73: (0, 0, 0, 255),
        74: (181, 112, 91, 255),
        75: (0, 165, 130, 255),
        76: (233, 214, 175, 255),
        77: (177, 154, 112, 255),
        78: (0, 0, 0, 255),
        79: (0, 0, 0, 255),
        80: (0, 0, 0, 255),
        81: (242, 242, 242, 255),
        82: (154, 154, 154, 255),
        83: (75, 112, 163, 255),
        84: (0, 0, 0, 255),
        85: (0, 0, 0, 255),
        86: (0, 0, 0, 255),
        87: (126, 177, 177, 255),
        88: (232, 255, 191, 255),
        89: (0, 0, 0, 255),
        90: (0, 0, 0, 255),
        91: (0, 0, 0, 255),
        92: (0, 255, 255, 255),
        93: (0, 0, 0, 255),
        94: (0, 0, 0, 255),
        95: (0, 0, 0, 255),
        96: (0, 0, 0, 255),
        97: (0, 0, 0, 255),
        98: (0, 0, 0, 255),
        99: (0, 0, 0, 255),
        100: (0, 0, 0, 255),
        101: (0, 0, 0, 255),
        102: (0, 0, 0, 255),
        103: (0, 0, 0, 255),
        104: (0, 0, 0, 255),
        105: (0, 0, 0, 255),
        106: (0, 0, 0, 255),
        107: (0, 0, 0, 255),
        108: (0, 0, 0, 255),
        109: (0, 0, 0, 255),
        110: (0, 0, 0, 255),
        111: (75, 112, 163, 255),
        112: (211, 226, 249, 255),
        113: (0, 0, 0, 255),
        114: (0, 0, 0, 255),
        115: (0, 0, 0, 255),
        116: (0, 0, 0, 255),
        117: (0, 0, 0, 255),
        118: (0, 0, 0, 255),
        119: (0, 0, 0, 255),
        120: (0, 0, 0, 255),
        121: (154, 154, 154, 255),
        122: (154, 154, 154, 255),
        123: (154, 154, 154, 255),
        124: (154, 154, 154, 255),
        125: (0, 0, 0, 255),
        126: (0, 0, 0, 255),
        127: (0, 0, 0, 255),
        128: (0, 0, 0, 255),
        129: (0, 0, 0, 255),
        130: (0, 0, 0, 255),
        131: (204, 191, 163, 255),
        132: (0, 0, 0, 255),
        133: (0, 0, 0, 255),
        134: (0, 0, 0, 255),
        135: (0, 0, 0, 255),
        136: (0, 0, 0, 255),
        137: (0, 0, 0, 255),
        138: (0, 0, 0, 255),
        139: (0, 0, 0, 255),
        140: (0, 0, 0, 255),
        141: (147, 204, 147, 255),
        142: (147, 204, 147, 255),
        143: (147, 204, 147, 255),
        144: (0, 0, 0, 255),
        145: (0, 0, 0, 255),
        146: (0, 0, 0, 255),
        147: (0, 0, 0, 255),
        148: (0, 0, 0, 255),
        149: (0, 0, 0, 255),
        150: (0, 0, 0, 255),
        151: (0, 0, 0, 255),
        152: (198, 214, 158, 255),
        153: (0, 0, 0, 255),
        154: (0, 0, 0, 255),
        155: (0, 0, 0, 255),
        156: (0, 0, 0, 255),
        157: (0, 0, 0, 255),
        158: (0, 0, 0, 255),
        159: (0, 0, 0, 255),
        160: (0, 0, 0, 255),
        161: (0, 0, 0, 255),
        162: (0, 0, 0, 255),
        163: (0, 0, 0, 255),
        164: (0, 0, 0, 255),
        165: (0, 0, 0, 255),
        166: (0, 0, 0, 255),
        167: (0, 0, 0, 255),
        168: (0, 0, 0, 255),
        169: (0, 0, 0, 255),
        170: (0, 0, 0, 255),
        171: (0, 0, 0, 255),
        172: (0, 0, 0, 255),
        173: (0, 0, 0, 255),
        174: (0, 0, 0, 255),
        175: (0, 0, 0, 255),
        176: (232, 255, 191, 255),
        177: (0, 0, 0, 255),
        178: (0, 0, 0, 255),
        179: (0, 0, 0, 255),
        180: (0, 0, 0, 255),
        181: (0, 0, 0, 255),
        182: (0, 0, 0, 255),
        183: (0, 0, 0, 255),
        184: (0, 0, 0, 255),
        185: (0, 0, 0, 255),
        186: (0, 0, 0, 255),
        187: (0, 0, 0, 255),
        188: (0, 0, 0, 255),
        189: (0, 0, 0, 255),
        190: (126, 177, 177, 255),
        191: (0, 0, 0, 255),
        192: (0, 0, 0, 255),
        193: (0, 0, 0, 255),
        194: (0, 0, 0, 255),
        195: (126, 177, 177, 255),
        196: (0, 0, 0, 255),
        197: (0, 0, 0, 255),
        198: (0, 0, 0, 255),
        199: (0, 0, 0, 255),
        200: (0, 0, 0, 255),
        201: (0, 0, 0, 255),
        202: (0, 0, 0, 255),
        203: (0, 0, 0, 255),
        204: (0, 255, 140, 255),
        205: (214, 158, 188, 255),
        206: (255, 102, 102, 255),
        207: (255, 102, 102, 255),
        208: (255, 102, 102, 255),
        209: (255, 102, 102, 255),
        210: (255, 142, 170, 255),
        211: (51, 73, 51, 255),
        212: (228, 112, 38, 255),
        213: (255, 102, 102, 255),
        214: (255, 102, 102, 255),
        215: (102, 153, 76, 255),
        216: (255, 102, 102, 255),
        217: (177, 154, 112, 255),
        218: (255, 142, 170, 255),
        219: (255, 102, 102, 255),
        220: (255, 142, 170, 255),
        221: (255, 102, 102, 255),
        222: (255, 102, 102, 255),
        223: (255, 142, 170, 255),
        224: (0, 175, 75, 255),
        225: (255, 211, 0, 255),
        226: (255, 211, 0, 255),
        227: (255, 102, 102, 255),
        228: (255, 210, 0, 255),
        229: (255, 102, 102, 255),
        230: (137, 98, 84, 255),
        231: (255, 102, 102, 255),
        232: (255, 38, 38, 255),
        233: (226, 0, 124, 255),
        234: (255, 158, 11, 255),
        235: (255, 158, 11, 255),
        236: (165, 112, 0, 255),
        237: (255, 211, 0, 255),
        238: (165, 112, 0, 255),
        239: (38, 112, 0, 255),
        240: (38, 112, 0, 255),
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
        251: (0, 0, 0, 255),
        252: (0, 0, 0, 255),
        253: (0, 0, 0, 255),
        254: (38, 112, 0, 255),
        255: (0, 0, 0, 255),
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
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
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(root, crs, res, transforms, cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            if not fname.endswith(".zip"):
                return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
        if glob.glob(pathname):
            self._extract()
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
        for year, md5 in self.md5s:
            download_url(
                self.url.format(year), self.root, md5=md5 if self.checksum else None
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)

    def plot(
        self,
        sample: Dict[str, Any],
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

        cmap: "np.typing.NDArray[np.int_]" = np.array(
            [self.cmap[i] for i in range(len(self.cmap))]
        )
        mask = cmap[mask]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze().numpy()
            pred = cmap[pred]
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return
