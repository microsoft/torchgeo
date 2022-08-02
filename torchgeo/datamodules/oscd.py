# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD datamodule."""

from typing import Any, Dict, List, Optional, Tuple

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Compose, Normalize

from ..datasets import OSCD
from .utils import dataset_split


class OSCDDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the OSCD dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.2
    """

    band_means = torch.tensor(
        [
            1583.0741,
            1374.3202,
            1294.1616,
            1325.6158,
            1478.7408,
            1933.0822,
            2166.0608,
            2076.4868,
            2306.0652,
            690.9814,
            16.2360,
            2080.3347,
            1524.6930,
        ]
    )

    band_stds = torch.tensor(
        [
            52.1937,
            83.4168,
            105.6966,
            151.1401,
            147.4615,
            115.9289,
            123.1974,
            114.6483,
            141.4530,
            73.2758,
            4.8368,
            213.4821,
            179.4793,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        bands: str = "all",
        train_batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        patch_size: Tuple[int, int] = (64, 64),
        num_patches_per_tile: int = 32,
        pad_size: Tuple[int, int] = (1280, 1280),
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for OSCD based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the OSCD Dataset classes
            bands: "rgb" or "all"
            train_batch_size: The batch size used in the train DataLoader
                (val_batch_size == test_batch_size == 1)
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            patch_size: Size of random patch from image and mask (height, width)
            num_patches_per_tile: number of random patches per sample
            pad_size: size to pad images to during val/test steps
        """
        super().__init__()
        self.root_dir = root_dir
        self.bands = bands
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.patch_size = patch_size
        self.num_patches_per_tile = num_patches_per_tile

        if bands == "rgb":
            self.band_means = self.band_means[[3, 2, 1]]
            self.band_stds = self.band_stds[[3, 2, 1]]

        self.rcrop = K.AugmentationSequential(
            K.RandomCrop(patch_size), data_keys=["input", "mask"], same_on_batch=True
        )
        self.padto = K.PadTo(pad_size)

        self.norm = Normalize(self.band_means, self.band_stds)

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()
        sample["image"] = self.norm(sample["image"])
        sample["image"] = torch.flatten(sample["image"], 0, 1)
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        OSCD(self.root_dir, split="train", bands=self.bands, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """

        def n_random_crop(sample: Dict[str, Any]) -> Dict[str, Any]:
            images, masks = [], []
            for i in range(self.num_patches_per_tile):
                mask = repeat(sample["mask"], "h w -> t h w", t=2).float()
                image, mask = self.rcrop(sample["image"], mask)
                mask = mask.squeeze()[0]
                images.append(image.squeeze())
                masks.append(mask.long())
            sample["image"] = torch.stack(images)
            sample["mask"] = torch.stack(masks)
            return sample

        def pad_to(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample["image"] = self.padto(sample["image"])[0]
            sample["mask"] = self.padto(sample["mask"].float()).long()[0, 0]
            return sample

        train_transforms = Compose([self.preprocess, n_random_crop])
        # for testing and validation we pad all inputs to a fixed size to avoid issues
        # with the upsampling paths in encoder-decoder architectures
        test_transforms = Compose([self.preprocess, pad_to])

        train_dataset = OSCD(
            self.root_dir, split="train", bands=self.bands, transforms=train_transforms
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            val_dataset = OSCD(
                self.root_dir,
                split="train",
                bands=self.bands,
                transforms=test_transforms,
            )
            self.train_dataset, self.val_dataset, _ = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
            self.val_dataset.dataset = val_dataset
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = OSCD(
            self.root_dir, split="test", bands=self.bands, transforms=test_transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""

        def collate_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            r_batch: Dict[str, Any] = default_collate(  # type: ignore[no-untyped-call]
                batch
            )
            r_batch["image"] = torch.flatten(r_batch["image"], 0, 1)
            r_batch["mask"] = torch.flatten(r_batch["mask"], 0, 1)
            return r_batch

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )
