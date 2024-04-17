# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datamodules."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import SpaceNet1
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class SpaceNet1DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SpaceNet1 dataset.

    Randomly splits into train/val/test.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SpaceNet1DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SpaceNet1`.
        """
        super().__init__(SpaceNet1, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.PadTo((448, 448)),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.PadTo((448, 448)),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SpaceNet1(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                1 - self.val_split_pct - self.test_split_pct,
                self.val_split_pct,
                self.test_split_pct,
            ],
            generator,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        batch["mask"] += 1

        return super().on_after_batch_transfer(batch, dataloader_idx)
