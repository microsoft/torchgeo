# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import PASTIS
from .geo import NonGeoDataModule

class PastisDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS dataset.

    .. versionadded:: 0.8
    """

    def __init__(self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any) -> None:
        """Initialize a new PastisDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS`.
        """
        super().__init__(PASTIS, batch_size=batch_size, num_workers=num_workers, **kwargs)
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
            self.train_dataset = PASTIS(split='train', **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = PASTIS(split='val', **self.kwargs)
        if stage in ['predict']:
            self.predict_dataset = PASTIS(split='test', **self.kwargs)