# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SKy Images and Photovoltaic Power Dataset (SKIPP'D) datamodule."""

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from ..datasets import SKIPPD
from .geo import NonGeoDataModule


class SKIPPDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SKIPP'd dataset.

    Implements 80/20 train/val splits on train_val set.
    See :func:`setup` for more details.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SkippdDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SKIPPD`.
        """
        super().__init__(SKIPPD, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = SKIPPD(split="trainval", **self.kwargs)

            train_indices, val_indices = train_test_split(
                np.arange(len(self.dataset)), test_size=0.2, train_size=0.8
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
        if stage in ["test"]:
            self.test_dataset = SKIPPD(split="test", **self.kwargs)
