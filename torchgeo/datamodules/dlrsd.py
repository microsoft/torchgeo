# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""DLRSD datamodules."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import DLRSD, DLRSDBase, DLRSDMultilabel
from .geo import NonGeoDataModule


class DLRSDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the DLRSD dataset.

    Uses random train/val/test splits.

    .. versionadded:: 0.10
    """

    std = torch.tensor(255)

    _dataset_cls: type[DLRSDBase] = DLRSD

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DLRSDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Fraction of dataset to use for validation. If 0, the
                train set is also used for validation.
            test_split_pct: Fraction of dataset to use for testing.
            seed: Random seed for reproducible train/val/test splits.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DLRSD`.
        """
        super().__init__(self._dataset_cls, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.seed = seed

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = self._dataset_cls(**self.kwargs)
        generator = torch.Generator().manual_seed(self.seed)

        if self.val_split_pct > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset,
                [
                    1 - self.val_split_pct - self.test_split_pct,
                    self.val_split_pct,
                    self.test_split_pct,
                ],
                generator,
            )
        else:
            self.train_dataset, self.test_dataset = random_split(
                dataset, [1 - self.test_split_pct, self.test_split_pct], generator
            )
            self.val_dataset = self.train_dataset


class DLRSDMultilabelDataModule(DLRSDDataModule):
    """LightningDataModule implementation for the DLRSDMultilabel dataset.

    Uses random train/val/test splits.

    .. versionadded:: 0.10
    """

    _dataset_cls: type[DLRSDBase] = DLRSDMultilabel
