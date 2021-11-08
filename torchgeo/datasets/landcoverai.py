# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai dataset."""

import hashlib
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from .geo import VisionDataset
from .utils import check_integrity, download_and_extract_archive, working_dir

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class LandCoverAI(VisionDataset):
    r"""LandCover.ai dataset.

    The `LandCover.ai <https://landcover.ai/>`_ (Land Cover from Aerial Imagery)
    dataset is a dataset for automatic mapping of buildings, woodlands, water and
    roads from aerial images. This implementation is specifically for Version 1 of
    Landcover.ai.

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

        Args:
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
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        with open(os.path.join(self.root, split + ".txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = self.ids[index].rstrip()
        sample = {"image": self._load_image(id_), "mask": self._load_target(id_)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    @lru_cache()
    def _load_image(self, id_: str) -> Tensor:
        """Load a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = os.path.join(self.root, "output", id_ + ".jpg")
        with Image.open(filename) as img:
            array = np.array(img)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    @lru_cache()
    def _load_target(self, id_: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the target mask
        """
        filename = os.path.join(self.root, "output", id_ + "_m.png")
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
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )

        return integrity

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

        # Generate train/val/test splits
        # Always check the sha256 of this file before executing
        # to avoid malicious code injection
        with working_dir(self.root):
            with open("split.py") as f:
                split = f.read().encode("utf-8")
                assert hashlib.sha256(split).hexdigest() == self.sha256
                exec(split)


class LandCoverAIDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the LandCover.ai dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self, root_dir: str, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for LandCover.ai based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the Landcover.AI Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"] / 255.0

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].float().unsqueeze(0) + 1

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        _ = LandCoverAI(self.root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_transforms = self.preprocess
        val_test_transforms = self.preprocess

        self.train_dataset = LandCoverAI(
            self.root_dir, split="train", transforms=train_transforms
        )

        self.val_dataset = LandCoverAI(
            self.root_dir, split="val", transforms=val_test_transforms
        )

        self.test_dataset = LandCoverAI(
            self.root_dir, split="test", transforms=val_test_transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
