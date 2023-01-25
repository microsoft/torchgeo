# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import LandCoverAI
from ..transforms import AugmentationSequential
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

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(
                p=0.5,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                silence_instantiation_warning=True,
            ),
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )
