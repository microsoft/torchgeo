# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai datamodules."""

from typing import Any

import kornia.augmentation as K

from ..datasets import LandCoverAI, LandCoverAI100
from .geo import NonGeoDataModule


class LandCoverAIDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LandCover.ai dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LandCoverAIDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LandCoverAI`.
        """
        super().__init__(LandCoverAI, batch_size, num_workers, **kwargs)

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=None,
            keepdim=True,
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )

        # https://github.com/kornia/kornia/issues/2848
        self.train_aug.keepdim = True
        self.aug.keepdim = True


class LandCoverAI100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LandCoverAI100 dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.7
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LandCoverAI100DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LandCoverAI100`.
        """
        super().__init__(LandCoverAI100, batch_size, num_workers, **kwargs)

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )

        # https://github.com/kornia/kornia/issues/2848
        self.aug.keepdim = True
