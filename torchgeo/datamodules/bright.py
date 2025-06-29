# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BRIGHT datamodule."""

from typing import Any

import kornia.augmentation as K
from torch import Tensor

from ..datasets import BRIGHTDFC2025
from .geo import NonGeoDataModule


class BRIGHTDFC2025DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BRIGHT dataset.

    .. versionadded:: 0.8
    """

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new BRIGHTBRIGHTDFC2025DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.BRIGHTDFC2025`.
        """
        super().__init__(
            BRIGHTDFC2025, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = BRIGHTDFC2025(split='train', **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = BRIGHTDFC2025(split='val', **self.kwargs)
        if stage in ['predict']:
            # Test set labels are not publicly available
            self.predict_dataset = BRIGHTDFC2025(split='test', **self.kwargs)

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
        if 'mask' in batch and batch['mask'].ndim == 3:
            batch['mask'] = batch['mask'].unsqueeze(dim=0)

        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        if 'mask' in batch:
            batch['mask'] = batch['mask'].squeeze(dim=1)
        return batch
