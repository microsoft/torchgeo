# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import RESISC45
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class RESISC45DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    mean = torch.tensor([127.86820969, 127.88083247, 127.84341029])
    std = torch.tensor([51.8668062, 47.2380768, 47.0613924])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new RESISC45DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.RESISC45`.
        """
        super().__init__(RESISC45, batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["image"],
        )
