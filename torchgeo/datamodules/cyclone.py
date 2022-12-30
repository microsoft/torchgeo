# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tropical Cyclone Wind Estimation Competition datamodule."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from kornia.augmentation import Normalize
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from ..datasets import TropicalCyclone
from ..transforms import AugmentationSequential


class TropicalCycloneDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.

    .. versionchanged:: 0.4
        Class name changed from CycloneDataModule to TropicalCycloneDataModule to be
        consistent with TropicalCyclone dataset.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for NASA Cyclone based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.TropicalCyclone`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.transform = AugmentationSequential(
            Normalize(mean=0, std=255), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Initialize the main Dataset objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        if self.kwargs.get("download", False):
            TropicalCyclone(split="train", **self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        This method is called once per GPU per run.

        We split samples between train/val by the ``storm_id`` property. I.e. all
        samples with the same ``storm_id`` value will be either in the train or the val
        split. This is important to test one type of generalizability -- given a new
        storm, can we predict its windspeed. The test set, however, contains *some*
        storms from the training set (specifically, the latter parts of the storms) as
        well as some novel storms.

        Args:
            stage: stage to set up
        """
        self.all_train_dataset = TropicalCyclone(split="train", **self.kwargs)

        storm_ids = []
        for item in self.all_train_dataset.collection:
            storm_id = item["href"].split("/")[0].split("_")[-2]
            storm_ids.append(storm_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2).split(
                storm_ids, groups=storm_ids
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)
        self.test_dataset = TropicalCyclone(split="test", **self.kwargs)

    def train_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
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

    def val_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
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

    def test_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
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
        """Run :meth:`torchgeo.datasets.TropicalCyclone.plot`."""
        return self.test_dataset.plot(*args, **kwargs)
