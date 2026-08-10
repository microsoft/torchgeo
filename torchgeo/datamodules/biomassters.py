# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""BioMassters datamodule."""

from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import torch
from torch.utils.data import random_split

from ..datasets import BioMassters
from ..datasets.utils import pad_across_batches
from .geo import NonGeoDataModule


class BioMasstersDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BioMassters dataset.

    Samples have the fused spatiotemporal regression format
    ``{'image': (T, C, H, W), 'mask': (H, W)}``.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        padding_length: int = 12,
        sensors: Sequence[Literal['S1', 'S2']] = ('S1', 'S2'),
        **kwargs: Any,
    ) -> None:
        """Initialize a new BioMasstersDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the labeled train split used for validation.
            test_split_pct: Percentage of the labeled train split used for testing.
            padding_length: Padding length of the time series.
            sensors: Sensors to include in the fused time series.
            **kwargs: Additional keyword arguments passed to the dataset.
        """
        super().__init__(
            BioMassters,
            batch_size=batch_size,
            num_workers=num_workers,
            sensors=sensors,
            as_time_series=True,
            **kwargs,
        )
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.padding_length = padding_length
        self.collate_fn = partial(
            pad_across_batches, padding_length=self.padding_length
        )
        self.aug = torch.nn.Identity()

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate', 'test']:
            self.dataset = BioMassters(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [
                    1 - self.val_split_pct - self.test_split_pct,
                    self.val_split_pct,
                    self.test_split_pct,
                ],
                generator,
            )

        if stage in ['predict']:
            self.predict_dataset = BioMassters(split='test', **self.kwargs)
