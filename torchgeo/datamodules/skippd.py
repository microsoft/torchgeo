# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SKy Images and Photovoltaic Power Dataset (SKIPP'D) datamodule."""

from typing import Any

import torch
from torch.utils.data import random_split

from ..datasets import SKIPPD
from .geo import NonGeoDataModule


class SKIPPDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SKIPP'D dataset.

    Implements 80/20 train/val splits on train_val set.
    See :func:`setup` for more details.

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SKIPPDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SKIPPD`.
        """
        super().__init__(SKIPPD, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = SKIPPD(split="trainval", **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ["test"]:
            self.test_dataset = SKIPPD(split="test", **self.kwargs)
