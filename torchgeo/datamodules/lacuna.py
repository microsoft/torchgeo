# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Lacuna African Field Boundaries datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import LacunaAfricanFieldBoundaries
from .geo import NonGeoDataModule


class LacunaAfricanFieldBoundariesDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Lacuna African Field Boundaries dataset.

    .. versionadded:: 0.8
    """

    mean = torch.tensor([0])
    std = torch.tensor([255])

    def __init__(
        self,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LacunaAfricanFieldBoundariesDataModule instance.

        Args:
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LacunaAfricanFieldBoundaries`.
        """
        super().__init__(
            LacunaAfricanFieldBoundaries, batch_size, num_workers, **kwargs
        )

        self.train_split_pct = 1 - val_split_pct - test_split_pct
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            data_keys=None,
            keepdim=True,
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', or 'test'.
        """
        if stage in ['fit', 'validate']:
            dataset = LacunaAfricanFieldBoundaries(**self.kwargs)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [self.train_split_pct, self.val_split_pct, self.test_split_pct]
            )
