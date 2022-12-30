# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT datamodule."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from kornia.augmentation import Normalize
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import EuroSAT
from ..transforms import AugmentationSequential


class EuroSATDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the EuroSAT dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    band_means = [
        1354.40546513,
        1118.24399958,
        1042.92983953,
        947.62620298,
        1199.47283961,
        1999.79090914,
        2369.22292565,
        2296.82608323,
        732.08340178,
        12.11327804,
        1819.01027855,
        1118.92391149,
        2594.14080798,
    ]

    band_stds = [
        245.71762908,
        333.00778264,
        395.09249139,
        593.75055589,
        566.4170017,
        861.18399006,
        1086.63139075,
        1117.98170791,
        404.91978886,
        4.77584468,
        1002.58768311,
        761.30323499,
        1231.58581042,
    ]

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.EuroSAT`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.transform = AugmentationSequential(
            Normalize(mean=self.band_means, std=self.band_stds), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            EuroSAT(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = EuroSAT(split="train", **self.kwargs)
        self.val_dataset = EuroSAT(split="val", **self.kwargs)
        self.test_dataset = EuroSAT(split="test", **self.kwargs)

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
        return self.transform(batch)

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.EuroSAT.plot`."""
        return self.val_dataset.plot(*args, **kwargs)
