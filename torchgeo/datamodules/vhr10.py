# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NWPU VHR-10 datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import VHR10
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.utils import AugPipe, collate_fn_detection
from .geo import NonGeoDataModule


class VHR10DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the VHR10 dataset.

    .. versionadded:: 0.6
    """

    std = torch.tensor(255)

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 512,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new VHR10DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.VHR10`.
        """
        super().__init__(VHR10, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = _to_tuple(patch_size)

        self.collate_fn = collate_fn_detection

        self.train_aug = AugPipe(
            AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(self.patch_size),
                K.RandomHorizontalFlip(),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.7),
                K.RandomVerticalFlip(),
                data_keys=["image", "boxes", "masks"],
            ),
            batch_size,
        )
        self.aug = AugPipe(
            AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(self.patch_size),
                data_keys=["image", "boxes", "masks"],
            ),
            batch_size,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = VHR10(**self.kwargs)
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
