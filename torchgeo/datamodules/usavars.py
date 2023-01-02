# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import USAVars
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class USAVarsDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the USAVars dataset.

    Uses random train/val/test splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new USAVarsDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.USAVars`.
        """
        super().__init__(USAVars, batch_size, num_workers, **kwargs)

        self.aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0), data_keys=["image"]
        )
