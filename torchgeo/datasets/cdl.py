"""CDL dataset."""

import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torch import Tensor

from .geo import GeoDataset
from .utils import BoundingBox, check_integrity, download_and_extract_archive

_crs = CRS.from_wkt(
    """
PROJCS["Albers Conical Equal Area",
    GEOGCS["NAD83",
        DATUM["North_American_Datum_1983",
            SPHEROID["GRS 1980",6378137,298.257222101,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6269"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4269"]],
    PROJECTION["Albers_Conic_Equal_Area"],
    PARAMETER["latitude_of_center",23],
    PARAMETER["longitude_of_center",-96],
    PARAMETER["standard_parallel_1",29.5],
    PARAMETER["standard_parallel_2",45.5],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["meters",1],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""
)


class CDL(GeoDataset):
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

    base_folder = "cdl"
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
        crs: CRS = _crs,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CDL Dataset.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = root
        self.crs = crs
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        fileglob = os.path.join(root, self.base_folder, "**_30m_cdls.img")
        for i, filename in enumerate(glob.iglob(fileglob)):
            year = int(os.path.basename(filename).split("_")[0])
            mint = datetime(year, 1, 1, 0, 0, 0).timestamp()
            maxt = datetime(year, 12, 31, 23, 59, 59).timestamp()
            with rasterio.open(filename) as src:
                cmap = src.colormap(1)
                with WarpedVRT(src, crs=self.crs) as vrt:
                    minx, miny, maxx, maxy = vrt.bounds
            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(i, coords, filename)
        self.cmap = np.array([cmap[i] for i in range(256)])

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} is not within bounds of the index: {self.bounds}"
            )

        hits = self.index.intersection(query, objects=True)
        filename = next(hits).object  # TODO: this assumes there is only a single hit
        with rasterio.open(filename) as src:
            with WarpedVRT(src, crs=self.crs) as vrt:
                window = rasterio.windows.from_bounds(
                    query.minx,
                    query.miny,
                    query.maxx,
                    query.maxy,
                    transform=vrt.transform,
                )
                masks = vrt.read(window=window)
        masks = masks.astype(np.int32)
        sample = {
            "masks": torch.tensor(masks),  # type: ignore[attr-defined]
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for year, md5 in self.md5s:
            filepath = os.path.join(
                self.root, self.base_folder, "{}_30m_cdls.zip".format(year)
            )
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for year, md5 in self.md5s:
            download_and_extract_archive(
                self.url.format(year),
                os.path.join(self.root, self.base_folder),
                md5=md5 if self.checksum else None,
            )

    def plot(self, image: Tensor) -> None:
        """Plot an image on a map.

        Args:
            image: the image to plot
        """
        # Convert from class labels to RGBA values
        array = image.squeeze().numpy()
        array = self.cmap[array]

        # Plot the image
        ax = plt.axes()
        ax.imshow(array, origin="lower")
        ax.axis("off")
        plt.show()
