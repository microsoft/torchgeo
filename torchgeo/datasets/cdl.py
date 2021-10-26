# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import download_url, extract_archive


class CDL(RasterDataset):
    """Cropland Data Layer (CDL) dataset.

    The `Cropland Data Layer
    <https://data.nal.usda.gov/dataset/cropscape-cropland-data-layer>`_, hosted on
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

    filename_glob = "*_30m_cdls.*"
    filename_regex = r"""
        ^(?P<date>\d+)
        _30m_cdls\..*$
    """
    zipfile_glob = "*_30m_cdls.zip"
    date_format = "%Y"
    is_image = False

    url = "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{}_30m_cdls.zip"  # noqa: E501
    md5s = [
        (2020, "97b3b5fd62177c9ed857010bca146f36"),
        (2019, "49d8052168c15c18f8b81ee21397b0bb"),
        (2018, "c7a3061585131ef049bec8d06c6d521e"),
        (2017, "dc8c1d7b255c9258d332dd8b23546c93"),
        (2016, "bb4df1b2ee6cedcc12a7e5a4527fcf1b"),
        (2015, "d17b4bb6ee7940af2c45d6854dafec09"),
        (2014, "6e0fcc800bd9f090f543104db93bead8"),
        (2013, "38df780d8b504659d837b4c53a51b3f7"),
        (2012, "2f3b46e6e4d91c3b7e2a049ba1531abc"),
        (2011, "dac7fe435c3c5a65f05846c715315460"),
        (2010, "18c9a00f5981d5d07ace69e3e33ea105"),
        (2009, "81a20629a4713de6efba2698ccb2aa3d"),
        (2008, "e6aa3967e379b98fd30c26abe9696053"),
    ]

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
                "to automaticaly download the dataset."
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
