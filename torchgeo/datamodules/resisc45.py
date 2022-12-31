# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 datamodule."""

from typing import Any, Dict, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import RESISC45
from ..transforms import AugmentationSequential


class RESISC45DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = [127.86820969, 127.88083247, 127.84341029]
    band_stds = [51.8668062, 47.2380768, 47.0613924]

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.RESISC45`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.train_transform = AugmentationSequential(
            K.Normalize(mean=self.band_means, std=self.band_stds),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(
                p=0.5,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                silence_instantiation_warning=True,
            ),
            data_keys=["image"],
        )
        self.test_transform = AugmentationSequential(
            K.Normalize(mean=self.band_means, std=self.band_stds), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            RESISC45(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = RESISC45(split="train", **self.kwargs)
        self.val_dataset = RESISC45(split="val", **self.kwargs)
        self.test_dataset = RESISC45(split="test", **self.kwargs)

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
        if self.trainer:
            if self.trainer.training:
                batch = self.train_transform(batch)
            elif self.trainer.validating or self.trainer.testing:
                batch = self.test_transform(batch)

        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.RESISC45.plot`.

        .. versionadded:: 0.2
        """
        return self.val_dataset.plot(*args, **kwargs)
