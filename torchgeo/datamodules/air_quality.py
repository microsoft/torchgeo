# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Air Quality datamodule."""

from typing import Any

import torch
from torch.utils.data import Subset

from ..datasets import AirQuality
from ..datasets.utils import Sample
from .geo import NonGeoDataModule


class AirQualityDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the AirQuality dataset.

    Uses the user provided splits to divide the dataset into
    train/val/test sets.

    .. versionadded:: 0.10
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

        window_size = dataset.input_steps + dataset.target_steps
        n = len(dataset) + window_size

        val_size = round(self.val_split_pct * n)
        test_size = round(self.test_split_pct * n)
        train_size = n - val_size - test_size

        # Be careful to avoid overlap between splits
        train_indices = range(train_size - window_size)
        val_indices = range(train_size, train_size + val_size - window_size)
        test_indices = range(train_size + val_size, n - window_size)

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

        # Compute normalization statistics from training data only
        input_data = torch.tensor(
            dataset.input_data.iloc[:train_size].values, dtype=torch.float32
        )
        target_data = torch.tensor(
            dataset.target_data.iloc[:train_size].values, dtype=torch.float32
        )

        self.input_mean = input_data.mean(dim=0)
        self.input_std = input_data.std(dim=0)
        self.target_mean = target_data.mean(dim=0)
        self.target_std = target_data.std(dim=0)

    def transfer_batch_to_device(
        self, batch: Sample, device: torch.device, dataloader_idx: int
    ) -> Sample:
        """Transfer batch and statistics to device.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.
        """
        self.input_mean = self.input_mean.to(device)
        self.input_std = self.input_std.to(device)
        self.target_mean = self.target_mean.to(device)
        self.target_std = self.target_std.to(device)

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Normalize batch data and pass normalization stats to the model.

        Overrides the base class to skip Kornia augmentations and instead
        apply dataset-level normalization to input and target using
        statistics computed from the training split.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data with normalized input and target.
        """
        batch['input'] = (batch['input'] - self.input_mean) / self.input_std
        batch['target'] = (batch['target'] - self.target_mean) / self.target_std
        return batch
