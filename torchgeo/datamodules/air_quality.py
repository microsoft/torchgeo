# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Air Quality datamodule."""

from typing import Any

import torch
from torch.utils.data import Subset

from ..datasets import AirQuality
from .geo import NonGeoDataModule


class AirQualityDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the AirQuality dataset.

    Uses the user provided splits to divide the dataset into
    train/val/test sets.

    .. versionadded:: 0.9
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
        self.mean: torch.Tensor = torch.tensor(0.0)
        self.std: torch.Tensor = torch.tensor(1.0)

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = AirQuality(**self.kwargs)

        window_size = dataset.num_past_steps + dataset.num_future_steps
        n = len(dataset)

        train_split_pct = 1 - (self.val_split_pct + self.test_split_pct)
        train_size = int(train_split_pct * n)
        val_size = int(self.val_split_pct * n)

        val_start = train_size + window_size
        test_start = val_start + val_size + window_size

        if test_start >= n:
            raise ValueError(
                f'Dataset too small ({n} samples) for the requested splits and '
                f'window size ({window_size}). Reduce num_past_steps, '
                f'num_future_steps, or the split percentages.'
            )

        train_indices = range(train_size)
        val_indices = range(val_start, val_start + val_size)
        test_indices = range(test_start, n)

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

        # Compute normalization stats from training data only.
        # train_size is derived from len(dataset) which is in sample index space,
        # but since sample index i corresponds to the window starting at raw row i,
        # iloc[:train_size] correctly selects only the raw rows covered by the training samples.
        train_data = torch.tensor(
            dataset.data.iloc[:train_size].values, dtype=torch.float32
        )
        self.mean = train_data.mean(dim=0)
        self.std = train_data.std(dim=0)

    def on_after_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Normalize batch data and pass normalization stats to the model.

        Overrides the base class to skip Kornia augmentations and instead
        apply dataset-level normalization to past and future targets using
        statistics computed from the training split.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data with normalized targets and normalization stats.
        """
        batch['past_targets'] = (batch['past_targets'] - self.mean) / (self.std + 1e-12)
        batch['future_targets'] = (batch['future_targets'] - self.mean) / (
            self.std + 1e-12
        )
        # Pass stats along so the model can unnormalize predictions before metric computation
        batch['mean'] = self.mean
        batch['std'] = self.std
        return batch
