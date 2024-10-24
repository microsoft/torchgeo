# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FTW datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import FieldsOfTheWorld
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class FieldsOfTheWorldDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FTW dataset.

    .. versionadded:: 0.7
    """

    mean = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    std = torch.tensor([3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new FTWDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FieldsOfTheWorld`.
        """
        super().__init__(FieldsOfTheWorld, batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            data_keys=['image', 'mask'],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=['image', 'mask']
        )
