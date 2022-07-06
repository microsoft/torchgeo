# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

import abc
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import PASTISInstanceSegmentation, PASTISSemanticSegmentation
from .utils import dataset_split


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Any]:
    """Custom object detection collate fn to handle variable number of boxes.

    Args:
        batch: list of sample dicts return by dataset
    Returns:
        batch dict output
    """
    output: Dict[str, Any] = {}
    output["image"] = torch.stack([sample["image"] for sample in batch])
    output["boxes"] = [sample["boxes"] for sample in batch]
    return output


class PASTISDataModule(pl.LightningDataModule, abc.ABC):
    """LightningDataModule implementation for the PASTIS dataset."""

    # (S1A, S1D, S2)
    band_means = torch.tensor(
        [
            -10.930951118469238,
            -17.348514556884766,
            6.417511940002441,
            -11.105852127075195,
            -17.502077102661133,
            6.407216548919678,
            1165.9398193359375,
            1375.6534423828125,
            1429.2191162109375,
            1764.798828125,
            2719.273193359375,
            3063.61181640625,
            3205.90185546875,
            3319.109619140625,
            2422.904296875,
            1639.370361328125,
        ]
    )
    band_stds = torch.tensor(
        [
            3.285966396331787,
            3.2129523754119873,
            3.3421084880828857,
            3.376193046569824,
            3.1705307960510254,
            3.34938383102417,
            1942.6156005859375,
            1881.9234619140625,
            1959.3798828125,
            1867.2239990234375,
            1754.5850830078125,
            1769.4046630859375,
            1784.860595703125,
            1767.7100830078125,
            1458.963623046875,
            1299.2833251953125,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for PASTIS based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the PASTIS Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        raise NotImplementedError

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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )


class PASTISSemanticSegmentationDataModule(PASTISDataModule):
    """LightningDataModule implementation for the PASTISSemanticSegmentation dataset."""

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        dataset = PASTISSemanticSegmentation(self.root_dir, transforms=transforms)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )


class PASTISInstanceSegmentationDataModule(PASTISDataModule):
    """LightningDataModule implementation for the PASTISInstanceSegmentation dataset."""

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        dataset = PASTISInstanceSegmentation(self.root_dir, transforms=transforms)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )
