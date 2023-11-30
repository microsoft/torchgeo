# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Digital Typhoon Data Module."""

from typing import Any

from torch.utils.data import Subset

from ..datasets import DigitalTyphoonAnalysis
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class DigitalTyphoonAnalysisDataModule(NonGeoDataModule):
    """Digital Typhoon Analysis Data Module."""

    valid_split_types = ["time", "id"]

    def __init__(
        self,
        split_type: str = "time",
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DigitalTyphoonAnalysisDataModule instance.

        Args:
            split_type: Either 'time' or 'id', which decides how to split the dataset for train, val, test
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DigitalTyphoonAnalysis`.

        """
        super().__init__(DigitalTyphoonAnalysis, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = DigitalTyphoonAnalysis(**self.kwargs)

        # TODO split into train, and test

        if stage in ["fit", "validate"]:
            # TODO split train into train and val

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
        if stage in ["test"]:
            self.test_dataset = DigitalTyphoonAnalysis(split="test", **self.kwargs)
