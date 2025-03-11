# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Air Quality datamodule."""

from typing import Any

from torch import Tensor
from torch.utils.data import Subset

from ..datasets import AirQuality
from .geo import NonGeoDataModule


class AirQualityDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the AirQuality dataset.

    Uses the user provided splits to divide the dataset into
    train/val/test sets.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        batch_size: int = 64,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new AirQualityDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a testing set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.AirQuality`.
        """
        super().__init__(AirQuality, batch_size, num_workers, **kwargs)
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = AirQuality(**self.kwargs)
        train_split_pct = 1 - (self.val_split_pct + self.test_split_pct)
        train_size = int(train_split_pct * len(dataset))
        val_size = int(self.val_split_pct * len(dataset))
        train_indices = range(train_size)
        val_indices = range(train_size, train_size + val_size)
        test_indices = range(train_size + val_size, len(dataset))
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

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
