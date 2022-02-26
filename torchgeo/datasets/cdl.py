# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CDL dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional, Tuple

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

    cmap: Dict[int, Tuple[int, int, int, int]] = {}

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
