# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""GeoNRW datamodule."""

import os
from typing import Any

import kornia.augmentation as K
from torch.utils.data import Subset

from ..datasets import GeoNRW
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class GeoNRWDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the GeoNRW dataset.

    Implements 80/20 train/val splits based on city locations.
    See :func:`setup` for more details.

    .. versionadded:: 0.6
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 256, **kwargs: Any
    ) -> None:
        """Initialize a new GeoNRWDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            size: resize images of input size 1000x1000 to size x size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.GeoNRW`.
        """
        super().__init__(GeoNRW, batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Resize(size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=['image', 'mask'],
        )

        self.aug = AugmentationSequential(K.Resize(size), data_keys=['image', 'mask'])

        self.size = size

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            dataset = GeoNRW(split='train', **self.kwargs)
            city_paths = [os.path.dirname(path) for path in dataset.file_list]
            train_indices, val_indices = group_shuffle_split(
                city_paths, test_size=0.2, random_state=0
            )
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
        if stage in ['test']:
            self.test_dataset = GeoNRW(split='test', **self.kwargs)
