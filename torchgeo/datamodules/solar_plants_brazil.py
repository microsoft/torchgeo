# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LightningDataModule for the SolarPlantsBrazil dataset."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import SolarPlantsBrazil
from ..samplers.utils import _to_tuple
from .geo import NonGeoDataModule

# Per-channel statistics (mean and std) computed only on the training split.
# Order corresponds to: [Red, Green, Blue, NIR]
MEAN = torch.tensor([927.7570, 740.1440, 492.3968, 2441.6775])
STD = torch.tensor([544.8361, 311.5538, 252.4914, 651.2599])


class SolarPlantsBrazilDataModule(NonGeoDataModule):
    """LightningDataModule for SolarPlantsBrazil dataset.

    This datamodule wraps the SolarPlantsBrazil dataset, which contains
    predefined train/val/test splits. This design ensures spatial separation
    between samples by solar plant, preventing data leakage during training.
    """

    def __init__(
        self,
        batch_size: int = 16,
        patch_size: tuple[int, int] | int = 256,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the SolarPlantsBrazilDataModule.

        Args:
            batch_size: Number of samples per batch.
            patch_size: Spatial dimensions (H, W) to crop from images.
            num_workers: Number of subprocesses used to load the data.
            **kwargs: Additional arguments passed to
                :class:`~torchgeo.datasets.SolarPlantsBrazil`.


        """
        super().__init__(
            dataset_class=SolarPlantsBrazil,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.patch_size = _to_tuple(patch_size)
        self.mean = MEAN
        self.std = STD

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(size=self.patch_size),
            data_keys=None,
            keepdim=True,
        )
