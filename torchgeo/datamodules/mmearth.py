# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MMEarth datamodule."""

from typing import Any

from torch import Tensor

from ..datasets import MMEarth
from .geo import NonGeoDataModule


class MMEarthDataModule(NonGeoDataModule):
    """LightningDataModule implementation for MMEarth dataset.

    Uses the train/val/test splits from the dataset. Normalization
    is handled in the dataset :class:`~torchgeo.datasets.MMEarth`.

    For additional Augmentation, use the `on_after_batch_transfer` method,
    and set the appropriate augmentations by inheriting from this class.

    .. versionadded:: 0.6
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new MMEarthDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.MMEarth`.
        """
        super().__init__(MMEarth, batch_size, num_workers, **kwargs)

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
        return batch
