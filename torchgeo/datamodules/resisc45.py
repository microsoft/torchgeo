# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import RESISC45
from .geo import NonGeoDataModule


class RESISC45DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    # Computed on the train set
    mean = torch.tensor([93.89391792, 97.11226906, 87.56775284])
    std = torch.tensor([51.84919672, 47.2365918, 47.06308786])

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

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=None,
        )
