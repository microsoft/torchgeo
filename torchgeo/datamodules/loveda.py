# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LoveDA datamodule."""

from typing import Any

from ..datasets import LoveDA
from .geo import NonGeoDataModule


class LoveDADataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LoveDA dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LoveDADataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LoveDA`.
        """
        super().__init__(LoveDA, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = LoveDA(split='train', **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = LoveDA(split='val', **self.kwargs)
        if stage in ['predict']:
            # Test set masks are not public, use for prediction instead
            self.predict_dataset = LoveDA(split='test', **self.kwargs)
