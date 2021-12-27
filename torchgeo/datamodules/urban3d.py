# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Urban3DChallenge datamodule."""

from typing import Any, Dict, Optional

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import Urban3DChallenge

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class Urban3DChallengeDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the Urban3DChallenge dataset."""

    # Global min/max band statistics computed on train set
    mins_raw = torch.tensor(  # type: ignore[attr-defined]
        [0.0, 0.0, 0.0, -32767.0, -32767.0, -11.262329]
    )[:, None, None]
    maxs_raw = torch.tensor(  # type: ignore[attr-defined]
        [255.0, 255.0, 255.0, 8.28125, 160.11902, 184.26416]
    )[:, None, None]

    # min/max band statistics percentile normalized to [2, 98]
    mins = torch.tensor(  # type: ignore[attr-defined]
        [0.0, 0.0, 0.0, -32767.0, -32767.0, -1.94026978]
    )[:, None, None]
    maxs = torch.tensor(  # type: ignore[attr-defined]
        [255.0, 255.0, 255.0, -0.108557129, 76.7863794, 103.39953]
    )[:, None, None]

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size: int = 512,
        augmentations: Optional[K.AugmentationSequential] = DEFAULT_AUGS,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Urban3DChallenge based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the Urban3D Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of random crops during train/val
            augmentations: augmentations applied to image and mask during train/val
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.random_crop = K.AugmentationSequential(
            K.RandomCrop(self.patch_size, p=1.0), data_keys=["input", "mask"]
        )

    def on_before_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform augmentations (if any) on batch before transfer to device.

        Args:
            batch: batch dict of samples
            dataloader_idx: current batch idx

        Returns:
            batch: batch dict of samples
        """
        if self.trainer.training or self.trainer.validating:
            batch["image"], batch["mask"] = self.random_crop(
                batch["image"], batch["mask"]
            )
        return batch

    def on_after_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform augmentations (if any) on batch after transfer to device.

        Args:
            batch: batch dict of samples
            dataloader_idx: current batch idx

        Returns:
            batch: batch dict of samples
        """
        if self.trainer.training or self.trainer.validating:
            if self.augmentations is not None:
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
        return batch

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = (sample["image"] - self.mins) / (self.maxs - self.mins)
        sample["image"] = torch.clamp(  # type: ignore[attr-defined]
            sample["image"], min=0.0, max=1.0
        )
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        self.train_dataset = Urban3DChallenge(
            self.root_dir, split="train", transforms=transforms
        )
        self.val_dataset = Urban3DChallenge(
            self.root_dir, split="val", transforms=transforms
        )
        self.test_dataset = Urban3DChallenge(
            self.root_dir, split="test", transforms=transforms
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
            self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )
