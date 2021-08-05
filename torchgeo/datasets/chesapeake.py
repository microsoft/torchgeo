"""Chesapeake Bay High-Resolution Land Cover Project dataset."""

import abc
import os
from typing import Any, Callable, Dict, Optional

from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import check_integrity, download_and_extract_archive


class Chesapeake(RasterDataset, abc.ABC):
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

    is_image = False

    @property
    @abc.abstractmethod
    def base_folder(self) -> str:
        """Parent directory of dataset in URL."""

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

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to. Uses the CRS
                of the files by default
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        super().__init__(root, crs, transforms)

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
        8: (0, 0, 0, 0),
        9: (0, 0, 0, 0),
        10: (0, 0, 0, 0),
        11: (0, 0, 0, 0),
        12: (0, 0, 0, 0),
        13: (0, 0, 0, 0),
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
