import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from .utils import working_dir


class LandCoverAI(VisionDataset):
    """The `LandCover.ai <https://landcover.ai/>`_ (Land Cover from Aerial Imagery)
    dataset is a dataset for automatic mapping of buildings, woodlands, water and
    roads from aerial images.

    Dataset features
    ~~~~~~~~~~~~~~~~

    * land cover from Poland, Central Europe
    * three spectral bands - RGB
    * 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
    * 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
    * total area of 216.27 km:sup:`2`

    Dataset format
    ~~~~~~~~~~~~~~

    * rasters are three-channel GeoTiffs with EPSG:2180 spatial reference system
    * masks are single-channel GeoTiffs with EPSG:2180 spatial reference system

    Dataset classes
    ~~~~~~~~~~~~~~~

    1. building (1.85 km:sup:`2`)
    2. woodland (72.02 km:sup:`2`)
    3. water (13.15 km:sup:`2`)
    4. road (3.5 km:sup:`2`)

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2005.02264v3

    .. note::

       This dataset requires the following additional library to be installed:

       * `opencv-python <https://pypi.org/project/opencv-python/>`_ to generate
         the train/val/test split
    """

    base_folder = "landcoverai"
    url = "https://landcover.ai/download/landcover.ai.v1.zip"
    filename = "landcover.ai.v1.zip"
    md5 = "3268c89070e8734b4e91d531c0617e03"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
        download: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Parameters:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transform: a function/transform that takes in a PIL image and returns a
                transformed version
            target_transform: a function/transform that takes in the target and
                transforms it
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
        """
        assert split in ["train", "val", "test"]

        super().__init__(root, transforms, transform, target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        with open(os.path.join(self.root, self.base_folder, split + ".txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = self.ids[index].rstrip()
        image = self._load_image(id_)
        target = self._load_target(id_)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_image(self, id_: str) -> Image.Image:
        """Load a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the image
        """
        return Image.open(
            os.path.join(self.root, self.base_folder, "output", id_ + ".jpg")
        ).convert("RGB")

    def _load_target(self, id_: str) -> Image.Image:
        """Load the target mask for a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the target mask
        """
        return Image.open(
            os.path.join(self.root, self.base_folder, "output", id_ + "_m.png")
        ).convert("L")

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.filename),
            self.md5,
        )

        return integrity

    def download(self) -> None:
        """Download the dataset and extract it."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            os.path.join(self.root, self.base_folder),
            filename=self.filename,
            md5=self.md5,
        )

        # Generate train/val/test splits
        with working_dir(os.path.join(self.root, self.base_folder)):
            with open("split.py") as f:
                exec(f.read())
