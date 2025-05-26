# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LightningDataModule for the SolarPlantsBrazil dataset."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor

from ..datasets import SolarPlantsBrazil
from ..samplers.utils import _to_tuple
from .geo import NonGeoDataModule

# Per-channel statistics (mean and std) computed over the entire dataset.
# Order corresponds to: [Red, Green, Blue, NIR]
MEAN = torch.tensor([1002.38684, 760.54535, 506.91257, 2422.154])
STD = torch.tensor([546.4485, 307.97183, 242.15088, 626.66785])


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
            **kwargs: Additional arguments passed to the dataset.
        """
        super().__init__(
            SolarPlantsBrazil, batch_size=batch_size, num_workers=num_workers, **kwargs
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

    def setup(self, stage: str) -> None:
        """Set up the datasets for training, validation, or testing.

        This method loads the pre-split datasets provided by SolarPlantsBrazil.
        These splits are spatially disjoint and must not be reshuffled.

        Args:
            stage: One of "fit", "validate", or "test".
        """
        if stage in ['fit', 'validate']:
            self.train_dataset = SolarPlantsBrazil(split='train', **self.kwargs)
            self.val_dataset = SolarPlantsBrazil(split='val', **self.kwargs)

        if stage == 'test':
            self.test_dataset = SolarPlantsBrazil(split='test', **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.

        """
        # This solves a special case where if batch_size=1 the mask won't be stacked correctly
        if batch['mask'].ndim == 3:
            batch['mask'] = batch['mask'].unsqueeze(dim=0)

        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        batch['mask'] = batch['mask'].squeeze(dim=1)
        return batch
