"""Chesapeake Bay High-Resolution Land Cover Project dataset."""

import abc
import os
import sys
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
PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",
    GEOGCS["NAD83",
        DATUM["North_American_Datum_1983",
            SPHEROID["GRS 1980",6378137,298.257222101004,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6269"]],
        PRIMEM["Greenwich",0],
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
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""
)


class Chesapeake(GeoDataset, abc.ABC):
    """Abstract base class for all Chesapeake datasets.

    `Chesapeake Bay High-Resolution Land Cover Project
    <https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/land-cover-data-project/>`_
    dataset.

    This dataset was collected by the Chesapeake Conservancy's Conservation Innovation
    Center (CIC) in partnership with the University of Vermont and WorldView Solutions,
    Inc. It consists of one-meter resolution land cover information for the Chesapeake
    Bay watershed (~100,000 square miles of land).

    For more information, see:

    * `User Guide
      <https://chesapeakeconservancy.org/wp-content/uploads/2017/01/LandCover101Guide.pdf>`_
    * `Class Descriptions
      <https://chesapeakeconservancy.org/wp-content/uploads/2020/03/LC_Class_Descriptions.pdf>`_
    * `Accuracy Assessment
      <https://chesapeakeconservancy.org/wp-content/uploads/2017/01/Chesapeake_Conservancy_Accuracy_Assessment_Methodology.pdf>`_

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

    @property
    @abc.abstractmethod
    def base_folder(self) -> str:
        """Subdirectory to find/store dataset in."""

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """Filename to find/store dataset in."""

    @property
    @abc.abstractmethod
    def zipfile(self) -> str:
        """Name of zipfile in download URL."""

    @property
    @abc.abstractmethod
    def md5(self) -> str:
        """MD5 checksum to verify integrity of dataset."""

    @property
    def url(self) -> str:
        """URL to download dataset from."""
        url = "https://cicwebresources.blob.core.windows.net/chesapeakebaylandcover"
        url += f"/{self.base_folder}/{self.zipfile}"
        return url

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (0, 168, 132, 255),
        3: (38, 115, 0, 255),
        4: (76, 230, 0, 255),
        5: (163, 255, 115, 255),
        6: (255, 170, 0, 255),
        7: (255, 0, 0, 255),
        8: (156, 156, 156, 255),
        9: (0, 0, 0, 255),
        10: (115, 115, 0, 255),
        11: (230, 230, 0, 255),
        12: (255, 255, 115, 255),
        13: (197, 0, 255, 255),
        14: (0, 0, 0, 0),
        15: (0, 0, 0, 0),
    }

    def __init__(
        self,
        root: str,
        crs: CRS = _crs,
        transforms: Optional[Callable[[Any], Any]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Chesapeake dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = os.path.join(root, "chesapeake", self.base_folder)
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
        filename = os.path.join(self.root, self.filename)
        with rasterio.open(filename) as src:
            with rasterio.open(filename) as src:
                with WarpedVRT(src, crs=self.crs) as vrt:
                    minx, miny, maxx, maxy = vrt.bounds
        mint = 0
        maxt = sys.maxsize
        coords = (minx, maxx, miny, maxy, mint, maxt)
        self.index.insert(0, coords, filename)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve labels and metadata indexed by query.

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
        filename = next(hits).object
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
            True if dataset MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.zipfile),
            self.md5 if self.checksum else None,
        )
        return integrity

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfile,
            md5=self.md5,
        )

    def plot(self, image: Tensor) -> None:
        """Plot an image on a map.

        Args:
            image: the image to plot
        """
        # Convert from class labels to RGBA values
        cmap = np.array([self.cmap[i] for i in range(len(self.cmap))])
        array = image.squeeze().numpy()
        array = cmap[array]

        # Plot the image
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()


class Chesapeake7(Chesapeake):
    """Complete 7-class dataset.

    This version of the dataset is composed of 7 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Tree Canopy and Shrubs: All woody vegetation including trees and shrubs
    3. Low Vegetation: Plant material less than 2 meters in height including lawns
    4. Barren: Areas devoid of vegetation consisting of natural earthen material
    5. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    6. Impervious Roads: Impervious surfaces that are used for transportation
    7. Aberdeen Proving Ground: U.S. Army facility with no labels
    """

    base_folder = "BAYWIDE"
    filename = "Baywide_7class_20132014.tif"
    zipfile = "Baywide_7Class_20132014.zip"
    md5 = "61a4e948fb2551840b6557ef195c2084"

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (255, 170, 0, 255),
        5: (156, 156, 156, 255),
        6: (0, 0, 0, 255),
        7: (197, 0, 255, 255),
        14: (0, 0, 0, 0),
        15: (0, 0, 0, 0),
    }


class Chesapeake13(Chesapeake):
    """Complete 13-class dataset.

    This version of the dataset is composed of 13 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Wetlands: Low vegetation areas located along marine or estuarine regions
    3. Tree Canopy: Deciduous and evergreen woody vegetation over 3-5 meters in height
    4. Shrubland: Heterogeneous woody vegetation including shrubs and young trees
    5. Low Vegetation: Plant material less than 2 meters in height including lawns
    6. Barren: Areas devoid of vegetation consisting of natural earthen material
    7. Structures: Human-constructed objects made of impervious materials
    8. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    9. Impervious Roads: Impervious surfaces that are used for transportation
    10. Tree Canopy over Structures: Tree cover overlapping impervious structures
    11. Tree Canopy over Impervious Surfaces: Tree cover overlapping impervious surfaces
    12. Tree Canopy over Impervious Roads: Tree cover overlapping impervious roads
    13. Aberdeen Proving Ground: U.S. Army facility with no labels
    """

    base_folder = "BAYWIDE"
    filename = "Baywide_13Class_20132014.tif"
    zipfile = "Baywide_13Class_20132014.zip"
    md5 = "7e51118923c91e80e6e268156d25a4b9"


class ChesapeakeDC(Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    base_folder = "DC"
    filename = os.path.join("DC_11001", "DC_11001.img")
    zipfile = "DC_11001.zip"
    md5 = "ed06ba7570d2955e8857d7d846c53b06"


class ChesapeakeDE(Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    base_folder = "DE"
    filename = "DE_STATEWIDE.tif"
    zipfile = "_DE_STATEWIDE.zip"
    md5 = "5e12eff3b6950c01092c7e480b38e544"


class ChesapeakeMD(Chesapeake):
    """This subset of the dataset contains data only for Maryland."""

    base_folder = "MD"
    filename = "MD_STATEWIDE.tif"
    zipfile = "_MD_STATEWIDE.zip"
    md5 = "40c7cd697a887f2ffdb601b5c114e567"


class ChesapeakeNY(Chesapeake):
    """This subset of the dataset contains data only for New York."""

    base_folder = "NY"
    filename = "NY_STATEWIDE.tif"
    zipfile = "_NY_STATEWIDE.zip"
    md5 = "1100078c526616454ef2e508affda915"


class ChesapeakePA(Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    base_folder = "PA"
    filename = "PA_STATEWIDE.tif"
    zipfile = "_PA_STATEWIDE.zip"
    md5 = "20a2a857c527a4dbadd6beed8b47e5ab"


class ChesapeakeVA(Chesapeake):
    """This subset of the dataset contains data only for Virginia."""

    base_folder = "VA"
    filename = "CIC2014_VA_STATEWIDE.tif"
    zipfile = "_VA_STATEWIDE.zip"
    md5 = "6f2c97deaf73bb3e1ea9b21bd7a3fc8e"


class ChesapeakeWV(Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    base_folder = "WV"
    filename = "WV_STATEWIDE.tif"
    zipfile = "_WV_STATEWIDE.zip"
    md5 = "350621ea293651fbc557a1c3e3c64cc3"
