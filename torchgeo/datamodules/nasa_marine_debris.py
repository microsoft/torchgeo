# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Marine Debris datamodule."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from kornia.augmentation import Normalize
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import NASAMarineDebris
from ..transforms import AugmentationSequential
from .utils import dataset_split


class NASAMarineDebrisDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NASA Marine Debris dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NASAMarineDebris`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.kwargs = kwargs

        self.transform = AugmentationSequential(
            Normalize(mean=0, std=255), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            NASAMarineDebris(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.dataset = NASAMarineDebris(**self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )

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

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
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
        """Run :meth:`torchgeo.datasets.NASAMarineDebris.plot`.

        .. versionadded:: 0.4
        """
        return self.dataset.plot(*args, **kwargs)
