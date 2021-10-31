# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced trainer."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional
from torch.nn.modules import Conv2d, Linear
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from ..datasets import UCMerced
from ..datasets.utils import dataset_split
from .tasks import ClassificationTask

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class UCMercedClassificationTask(ClassificationTask):
    """LightningModule for training models on the UC Merced Dataset."""

    num_classes = 21


class UCMercedDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the UC Merced dataset.

    Uses random train/val/test splits.
    """

    band_means = torch.tensor([0, 0, 0])  # type: ignore[attr-defined]

    band_stds = torch.tensor([1, 1, 1])  # type: ignore[attr-defined]

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        unsupervised_mode: bool = False,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for UCMerced based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the UCMerced Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unsupervised_mode = unsupervised_mode

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.norm = Normalize(self.band_means, self.band_stds)

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        c, h, w = sample["image"].shape
        if h != 256 or w != 256:
            sample["image"] = torchvision.transforms.functional.resize(
                sample["image"], size=(256, 256)
            )
        sample["image"] = self.norm(sample["image"])
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        UCMerced(self.root_dir, download=True, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        if not self.unsupervised_mode:

            dataset = UCMerced(self.root_dir, transforms=transforms)
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
            )
        else:

            self.train_dataset = UCMerced(self.root_dir, transforms=transforms)
            self.val_dataset, self.test_dataset = None, None  # type: ignore[assignment]

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
        if self.unsupervised_mode or self.val_split_pct == 0:
            return self.train_dataloader()
        else:
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
        if self.unsupervised_mode or self.test_split_pct == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
