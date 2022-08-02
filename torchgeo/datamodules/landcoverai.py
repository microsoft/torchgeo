# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai datamodule."""

from typing import Any, Dict, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets import LandCoverAI

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class LandCoverAIDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the LandCover.ai dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self, root_dir: str, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for LandCover.ai based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the Landcover.AI Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        sample["image"] /= 255.0

        if "mask" in sample:
            sample["mask"] = sample["mask"].long() + 1

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        LandCoverAI(self.root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_transforms = self.preprocess
        val_test_transforms = self.preprocess

        self.train_dataset = LandCoverAI(
            self.root_dir, split="train", transforms=train_transforms
        )

        self.val_dataset = LandCoverAI(
            self.root_dir, split="val", transforms=val_test_transforms
        )

        self.test_dataset = LandCoverAI(
            self.root_dir, split="test", transforms=val_test_transforms
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
        """Run :meth:`torchgeo.datasets.LandCoverAI.plot`.

        .. versionadded:: 0.2
        """
        return self.val_dataset.plot(*args, **kwargs)
