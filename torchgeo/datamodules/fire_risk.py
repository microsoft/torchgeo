# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FireRisk datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import FireRisk
from .geo import NonGeoDataModule


class FireRiskDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FireRisk dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new FireRiskDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FireRisk`.
        """
        super().__init__(FireRisk, batch_size, num_workers, **kwargs)
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=None,
            keepdim=True,
        )
        # https://github.com/kornia/kornia/issues/2848
        self.train_aug.keepdim = True  # type: ignore[attr-defined]

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = FireRisk(split='train', **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = FireRisk(split='val', **self.kwargs)
