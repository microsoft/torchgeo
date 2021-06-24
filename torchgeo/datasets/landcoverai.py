import hashlib
import os
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from .geo import VisionDataset
from .utils import working_dir


class LandCoverAI(VisionDataset):
    r"""The `LandCover.ai <https://landcover.ai/>`_ (Land Cover from Aerial Imagery)
    dataset is a dataset for automatic mapping of buildings, woodlands, water and
    roads from aerial images.

    Dataset features:

    * land cover from Poland, Central Europe
    * three spectral bands - RGB
    * 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
    * 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
    * total area of 216.27 km\ :sup:`2`

    Dataset format:

    * rasters are three-channel GeoTiffs with EPSG:2180 spatial reference system
    * masks are single-channel GeoTiffs with EPSG:2180 spatial reference system

    Dataset classes:

    1. building (1.85 km\ :sup:`2`\ )
    2. woodland (72.02 km\ :sup:`2`\ )
    3. water (13.15 km\ :sup:`2`\ )
    4. road (3.5 km\ :sup:`2`\ )

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
    sha256 = "15ee4ca9e3fd187957addfa8f0d74ac31bc928a966f76926e11b3c33ea76daa1"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Parameters:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["train", "val", "test"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        with open(os.path.join(self.root, self.base_folder, split + ".txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = self.ids[index].rstrip()
        sample = {
            "image": self._load_image(id_),
            "mask": self._load_target(id_),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_image(self, id_: str) -> Tensor:
        """Load a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = os.path.join(self.root, self.base_folder, "output", id_ + ".jpg")
        with Image.open(filename) as img:
            array = np.array(img)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, id_: str) -> Tensor:
        """Load the target mask for a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the target mask
        """
        filename = os.path.join(self.root, self.base_folder, "output", id_ + "_m.png")
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.filename),
            self.md5 if self.checksum else None,
        )

        return integrity

    def download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            os.path.join(self.root, self.base_folder),
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

        # Generate train/val/test splits
        # Always check the sha256 of this file before executing
        # to avoid malicious code injection
        with working_dir(os.path.join(self.root, self.base_folder)):
            with open("split.py") as f:
                split = f.read().encode("utf-8")
                assert hashlib.sha256(split).hexdigest() == self.sha256
                exec(split)
