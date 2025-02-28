# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Substation datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import Substation
from .geo import NonGeoDataModule


class SubstationDataModule(NonGeoDataModule):
    """Substation Data Module with train-test split and transformations.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        size: int = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SubstationDataModule instance.

        Args:
            root: Path to the dataset directory.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for data loading.
            val_split_pct: Percentage of data to use for validation.
            test_split_pct: Percentage of data to use for testing.
            model_type: Type of model being used (e.g., 'swin' for specific channel selection).
            size: Size of the input images.

            **kwargs: Additional arguments passed to Substation.
        """
        super().__init__(Substation, batch_size, num_workers, **kwargs)
        self.root = root
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.train_aug = K.AugmentationSequential(
            K.Resize(size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )
        self.aug = K.AugmentationSequential(
            K.Resize(size), data_keys=None, keepdim=True
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = Substation(
            root=self.root,
            use_timepoints=True,
            mask_2d=False,
            download=True,
            checksum=False,
            **self.kwargs,
        )

        generator = torch.Generator().manual_seed(0)
        total_len = len(dataset)
        val_len = int(total_len * self.val_split_pct)
        test_len = int(total_len * self.test_split_pct)
        train_len = total_len - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_len, val_len, test_len], generator
        )
