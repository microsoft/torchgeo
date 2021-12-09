# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LoveDA dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from .geo import VisionDataset
from .utils import download_and_extract_archive

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class LoveDA(VisionDataset):
    """LoveDA dataset.

    The `LoveDA <https://github.com/Junjue-Wang/LoveDA>`_ datataset is a
    semantic segmentation dataset.

    Dataset features:

    * 2713 urban scene and 3274 rural scene HSR images, spatial resolution of 0.3m
    * image source is Google Earth platform
    * total of 166768 annotated objects from Nanjing, Changzhou and Wuhan cities
    * dataset comes with predefined train, validation, and test set
    * dataset differentiates between 'rural' and 'urban' images

    Dataset format:

    * images are three-channel pngs with dimension 1024x1024
    * segmentation masks are single-channel pngs

    Dataset classes:

    1. background
    2. building
    3. road
    4. water
    5. barren
    6. forest
    7. agriculture

    No-data regions assigned with 0 and should be ignored.

    If you use this dataset in your research, please cite the following paper:

    * <https://arxiv.org/abs/2110.08733>

    .. versionadded:: 0.2
    """

    scenes = ["urban", "rural"]
    splits = ["train", "val", "test"]

    info_dict = {
        "train": {
            "url": "https://drive.google.com/file/d/1xbnKVN5aRMl \
            pxISXgutzQO0hPT_b4lMi/view?usp=sharing",
            "filename": "Train.zip",
            "md5": "de2b196043ed9b4af1690b3f9a7d558f",
        },
        "val": {
            "url": "https://drive.google.com/file/d/1yTMfeon1Oc4 \
                ia9oCX7r5Yq4C39I0oO_b/view?usp=sharing",
            "filename": "Val.zip",
            "md5": "84cae2577468ff0b5386758bb386d31d",
        },
        "test": {
            "url": "https://drive.google.com/file/d/1ON7bWat7u9f \
                GV16stAosdmzMIpcydnVC/view?usp=sharing",
            "filename": "Test.zip",
            "md5": "a489be0090465e01fb067795d24e6b47",
        },
    }

    classes = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
        "no-data",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        scene: List[str] = ["urban", "rural"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LoveDA dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            scene: specify whether to load only 'urban', only 'rural' or both
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            AssertionError: if ``scene`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        print(split)
        assert split in self.splits
        assert set(scene).intersection(
            set(self.scenes)
        ), "The possible scenes are 'rural' and/or 'urban'"
        assert len(scene) <= 2, "There are no other scenes than 'rural' or 'urban'"

        self.root = root
        self.split = split
        self.scene = scene
        self.transforms = transforms
        self.checksum = checksum

        self.url = self.info_dict[self.split]["url"]
        self.filename = self.info_dict[self.split]["filename"]
        self.md5 = self.info_dict[self.split]["md5"]

        self.directory = os.path.join(self.root, split.capitalize())
        self.scene_paths = [
            os.path.join(self.directory, s.capitalize()) for s in self.scene
        ]

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found at root directory or corrupted. "
                + "You can use download=True to download it"
            )

        self.files = self._load_files(self.scene_paths, self.split)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample: image and mask at that index with image of dimension 3x1024x1024
                    and mask of dimension 1024x1024
        """
        files = self.files[index]
        image = self._load_image(files["image"])

        if self.split != "test":
            mask = self._load_target(files["mask"])
            sample = {"image": image, "mask": mask}
        else:
            sample = {"image": image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)

    def _load_files(self, scene_paths: List[str], split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            scene_paths: contains one or two paths, depending on whether user has
                         specified only 'rural', 'only 'urban' or both
            split: subset of dataset, one of [train, val, test]
        """
        images = []

        for s in scene_paths:
            images.extend(glob.glob(os.path.join(s, "images_png", "*.png")))

        images = sorted(images)

        if self.split != "test":
            masks = [image.replace("images_png", "masks_png") for image in images]
            files = [
                dict(image=image, mask=mask) for image, mask, in zip(images, masks)
            ]
        else:
            files = [dict(image=image) for image in images]

        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            tensor: the loaded image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load a single mask corresponding to image.

        Args:
            path: path to the mask

        Returns:
            tensor: the mask of the image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
            return tensor

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for s in self.scene_paths:
            if not os.path.exists(s):
                return False

        return True

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

    def plot(
        self, sample: Dict[str, Tensor], suptitle: Optional[str] = None
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure
        Returns:
            fig: a matplotlib Figure with the rendered sample
        """
        if self.split != "test":
            image, mask = sample["image"], sample["mask"]
            ncols = 2
        else:
            image = sample["image"]
            ncols = 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        if self.split != "test":
            axs[0].imshow(image.permute(1, 2, 0))
            axs[0].axis("off")
            axs[1].imshow(mask)
            axs[1].axis("off")
        else:
            axs.imshow(image.permute(1, 2, 0))
            axs.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class LoveDADataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the LoveDA dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        scene: List[str],
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for LoveDA based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to LoveDA Dataset classes
            scene: specify whether to load only 'urban', only 'rural' or both
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.scene = scene
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

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        _ = LoveDA(self.root_dir, scene=self.scene, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_transforms = self.preprocess
        val_test_transforms = self.preprocess

        self.train_dataset = LoveDA(
            self.root_dir, split="train", scene=self.scene, transforms=train_transforms
        )

        self.val_dataset = LoveDA(
            self.root_dir, split="val", scene=self.scene, transforms=val_test_transforms
        )

        self.test_dataset = LoveDA(
            self.root_dir,
            split="test",
            scene=self.scene,
            transforms=val_test_transforms,
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
