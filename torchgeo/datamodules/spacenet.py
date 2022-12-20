# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datamodules."""

from typing import Any, Dict, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets import SpaceNet1
from .utils import dataset_split

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class SpaceNet1DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the SpaceNet1 dataset.

    Randomly splits into train/val/test.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet1.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SpaceNet1`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.kwargs = kwargs

        self.padto = K.PadTo((448, 448))

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
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
        ):
            # Kornia expects masks to be floats with a channel dimension
            x = batch["image"]
            y = batch["mask"].float().unsqueeze(1)

            train_augmentations = K.AugmentationSequential(
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomSharpness(p=0.5),
                K.ColorJitter(
                    p=0.5,
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                    silence_instantiation_warning=True,
                ),
                data_keys=["input", "mask"],
            )
            x, y = train_augmentations(x, y)

            # torchmetrics expects masks to be longs without a channel dimension
            batch["image"] = x
            batch["mask"] = y.squeeze(1).long()

        return batch

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] = self.padto(sample["image"]).squeeze()
        sample["image"] /= 255.0

        if "mask" in sample:
            # We add 1 to the mask to map the current {background, building} labels to
            # the values {1, 2}. This is necessary because we add 0 padding to the
            # mask that we want to ignore in the loss function.
            sample["mask"] = self.padto(sample["mask"].float() + 1).squeeze()
            sample["mask"] = sample["mask"].long()
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        SpaceNet1(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.dataset = SpaceNet1(transforms=self.preprocess, **self.kwargs)
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

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.SpaceNet1.plot`."""
        return self.dataset.plot(*args, **kwargs)
