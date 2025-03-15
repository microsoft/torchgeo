# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ReforesTree datamodule."""
from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import ReforesTree
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _RandomNCrop

from .geo import NonGeoDataModule



class ReforesTreeDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the ReforesTree dataset.

    Implements 80/20 train/val splits.

    .. versionadded:: 0.7
        
    """

    def __init__(
        self, 
        batch_size: int = 64, 
        patch_size: tuple[int, int] | int = 64, 
        num_workers: int = 0, 
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2, 
        **kwargs: Any
    ) -> None:
        """Initialize a new ReforesTreeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ReforesTree`.
        """
        super().__init__(ReforesTree, 1, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = _to_tuple(patch_size)

        self.train_aug = K.AugmentationSequential(
            K.Resize(patch_size),
            K.Normalize(self.mean, self.std),
            _RandomNCrop(self.patch_size, batch_size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(patch_size),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = ReforesTree(**self.kwargs)
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
