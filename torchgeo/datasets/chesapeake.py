"""`Chesapeake Bay High-Resolution Land Cover Project
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

import abc
import os
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class _Chesapeake(VisionDataset, abc.ABC):
    """Abstract base class for all Chesapeake datasets."""

    @property
    @abc.abstractmethod
    def base_folder(self):
        """Subdirectory to find/store dataset in."""
        pass

    @property
    @abc.abstractmethod
    def filename(self):
        """Filename to find/store dataset in."""
        pass

    @property
    @abc.abstractmethod
    def zipfile(self):
        """Name of zipfile in download URL."""
        pass

    @property
    @abc.abstractmethod
    def md5(self):
        """MD5 checksum to verify integrity of dataset."""
        pass

    @property
    def url(self):
        """URL to download dataset from."""
        url = "https://cicwebresources.blob.core.windows.net/chesapeakebaylandcover"
        url += f"/{self.base_folder}/{self.zipfile}"
        return url

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
        download: bool = False,
    ) -> None:
        """Initialize a new Chesapeake dataset instance.

        Parameters:
            root: root directory where dataset can be found
            transform: a function/transform that takes in a numpy array and returns a
                transformed version
            target_transform: a function/transform that takes in the target and
                transforms it
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
        """
        root = os.path.join(root, "chesapeake")
        super().__init__(root, transforms, transform, target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return an index within the dataset.

        Parameters:
            idx: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(idx)
        target = self._load_target(idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_image(self, id: int) -> None:
        """Load a single image.

        Parameters:
            id: unique ID of the image

        Returns:
            the image
        """
        pass

    def _load_target(self, id: int) -> Any:
        """Load the annotations for a single image.

        Parameters:
            id: unique ID of the image

        Returns:
            the annotations
        """
        pass

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5s match, else False
        """
        return check_integrity(
            os.path.join(self.root, self.filename),
            self.md5,
        )

    def download(self) -> None:
        """Download the dataset and extract it."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            os.path.join(self.root, self.base_folder),
            self.filename,
            self.md5,
        )


class Chesapeake7(_Chesapeake):
    """This version of the dataset is composed of 7 classes:

    0. No Data: Background values
    1. Barren: Areas devoid of vegetation consisting of natural earthen material
    2. Impervious Roads: Impervious surfaces that are used for transportation
    3. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    4. Low Vegetation: Plant material less than 2 meters in height including lawns
    5. Tree Canopy: Deciduous and evergreen woody vegetation over 3-5 meters in height
    6. Tree Canopy over Impervious Surfaces: Tree cover overlapping impervious surfaces
    7. Water: All areas of open water including ponds, rivers, and lakes
    """

    # TODO: make sure these class numbers are correct
    base_folder = "BAYWIDE"
    filename = "Baywide_7Class_20132014.tif"
    zipfile = "Baywide_7Class_20132014.zip"
    md5 = ""


class Chesapeake13(_Chesapeake):
    """This version of the dataset is composed of 13 classes:

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


class ChesapeakeDC(_Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    base_folder = "DC"
    filename = os.path.join("DC_11001", "DC_11001.img")
    zipfile = "DC_11001.zip"
    md5 = "ed06ba7570d2955e8857d7d846c53b06"


class ChesapeakeDE(_Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    base_folder = "DE"
    filename = "DE_STATEWIDE.tif"
    zipfile = "_DE_STATEWIDE.zip"
    md5 = "5e12eff3b6950c01092c7e480b38e544"


class ChesapeakeMD(_Chesapeake):
    """This subset of the dataset contains data only for Maryland."""

    base_folder = "MD"
    filename = "MD_STATEWIDE.tif"
    zipfile = "_MD_STATEWIDE.zip"
    md5 = "40c7cd697a887f2ffdb601b5c114e567"


class ChesapeakeNY(_Chesapeake):
    """This subset of the dataset contains data only for New York."""

    base_folder = "NY"
    filename = "NY_STATEWIDE.tif"
    zipfile = "_NY_STATEWIDE.zip"
    md5 = "1100078c526616454ef2e508affda915"


class ChesapeakePA(_Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    base_folder = "PA"
    filename = "PA_STATEWIDE.tif"
    zipfile = "_PA_STATEWIDE.zip"
    md5 = ""


class ChesapeakeVA(_Chesapeake):
    """This subset of the dataset contains data only for Virginia."""

    base_folder = "VA"
    filename = "CIC2014_VA_STATEWIDE.tif"
    zipfile = "_VA_STATEWIDE.zip"
    md5 = "6f2c97deaf73bb3e1ea9b21bd7a3fc8e"


class ChesapeakeWV(_Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    base_folder = "WV"
    filename = "WV_STATEWIDE.tif"
    zipfile = "_WV_STATEWIDE.zip"
    md5 = "350621ea293651fbc557a1c3e3c64cc3"
