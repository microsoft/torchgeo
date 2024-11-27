# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import UCMerced
from .geo import NonGeoDataModule


class UCMercedDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the UC Merced dataset.

    Uses random train/val/test splits.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new UCMercedDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.UCMerced`.
        """
        super().__init__(UCMerced, batch_size, num_workers, **kwargs)

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=256),
            data_keys=None,
            keepdim=True,
        )
