# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""AgriFieldNet datamodule."""

from typing import Any

from ..datasets import AgriFieldNet
from .geo import NonGeoDataModule
from .utils import dataset_split
from ..samplers.utils import _to_tuple


class AgriFieldNetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the AgriFieldNet dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int = 256,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize a new AgriFieldNetDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.AgriFieldNetDataModule`.
        """
        super().__init__(AgriFieldNet, batch_size, num_workers, **kwargs)
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate", "test"]:
            self.dataset = AgriFieldNet(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset=self.dataset,
                val_pct=self.val_split_pct,
                test_pct=self.test_split_pct,
            )
        if stage in ["predict"]:
            self.predict_dataset = AgriFieldNet(split="test", **self.kwargs)
