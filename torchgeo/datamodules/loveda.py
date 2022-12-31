# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LoveDA datamodule."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from kornia.augmentation import Normalize
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import LoveDA
from ..transforms import AugmentationSequential


class LoveDADataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the LoveDA dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for LoveDA based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LoveDA`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.transform = AugmentationSequential(
            Normalize(mean=0, std=255), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            LoveDA(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = LoveDA(split="train", **self.kwargs)
        self.val_dataset = LoveDA(split="val", **self.kwargs)

        # Test set masks are not public, use for prediction instead
        self.predict_dataset = LoveDA(split="test", **self.kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for prediction.

        Returns:
            predict data loader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply augmentations to batch after transferring to GPU.

        Args:
            batch: A batch of data that needs to be altered or augmented
            dataloader_idx: The index of the dataloader to which the batch belongs

        Returns:
            A batch of data
        """
        batch = self.transform(batch)
        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.LoveDA.plot`.

        .. versionadded:: 0.4
        """
        return self.train_dataset.plot(*args, **kwargs)
