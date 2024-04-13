# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NAIPCluster datamodule."""

import os
from typing import Any, Dict, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from ..datasets import NAIPCluster

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class NAIPClusterDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NAIPCluster dataset.

    #TODO: describe what is going on here.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        num_clusters: int = 64,
        cluster_radius: int = 1,
        num_cluster_samples: int = 10000,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NAIPCluster based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the NAIPCluster Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            num_clusters: The number of clusters to use in the KMeans model
            cluster_radius: The radius to use when clustering pixels. E.g. a radius of 0
                will create a KMeans model that just uses the R, G, B, NIR values at a
                single pixel, while a radius of 1 will consider all the spectral values
                in a 3x3 window centered at a pixel.
            num_cluster_samples: Number of points used to fit the KMeans model
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        self.num_cluster_samples = num_cluster_samples

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Apply batch augmentations after batch is transferred to the device.

        Args:
            batch: mini-batch of data
            batch_idx: batch index

        Returns:
            augmented mini-batch
        """
        if (
            hasattr(self, "trainer")
            and hasattr(self.trainer, "training")
            and self.trainer.training  # type: ignore[union-attr]
        ):
            x = batch["image"]

            train_augmentations = K.AugmentationSequential(
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["input"],
            )
            x = train_augmentations(x)

            batch["image"] = x

        return batch

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        # TODO: update this once the NAIPCluster dataset is downloadable
        assert os.path.exists(self.root_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.dataset = NAIPCluster(
            self.root_dir,
            self.num_clusters,
            self.cluster_radius,
            self.num_cluster_samples,
            transform=self.preprocess,
        )

        train_indices = list(range(0, 40000))
        val_indices = list(range(40000, 45000))
        test_indices = list(range(45000, 50000))

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

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

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.NAIPCluster.plot`."""
        return self.dataset.plot(*args, **kwargs)
