# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ETCI 2021 datamodule."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Normalize

from ..datasets import ETCI2021


class ETCI2021DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the ETCI2021 dataset.

    Splits the existing train split from the dataset into train/val with 80/20
    proportions, then uses the existing val dataset as the test data.

    .. versionadded:: 0.2
    """

    band_means = torch.tensor(
        [0.52253931, 0.52253931, 0.52253931, 0.61221701, 0.61221701, 0.61221701]
    )

    band_stds = torch.tensor(
        [0.35221376, 0.35221376, 0.35221376, 0.37364622, 0.37364622, 0.37364622]
    )

    def __init__(
        self,
        root_dir: str,
        seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for ETCI2021 based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the ETCI2021 Dataset classes
            seed: The seed value to use when doing the dataset random_split
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.norm = Normalize(self.band_means, self.band_stds)

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Notably, moves the given water mask to act as an input layer.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        sample["image"] = self.norm(sample["image"])

        if "mask" in sample:
            flood_mask = sample["mask"][1]
            flood_mask = (flood_mask > 0).long()
            sample["mask"] = flood_mask

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        ETCI2021(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_val_dataset = ETCI2021(
            self.root_dir, split="train", transforms=self.preprocess
        )
        self.test_dataset = ETCI2021(
            self.root_dir, split="val", transforms=self.preprocess
        )

        size_train_val = len(train_val_dataset)
        size_train = round(0.8 * size_train_val)
        size_val = size_train_val - size_train

        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset,
            [size_train, size_val],
            generator=Generator().manual_seed(self.seed),
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
        """Run :meth:`torchgeo.datasets.ETCI2021.plot`."""
        return self.test_dataset.plot(*args, **kwargs)
