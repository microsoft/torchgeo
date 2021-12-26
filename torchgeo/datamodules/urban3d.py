# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""US3D datamodule."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import US3D

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class US3DDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the US3D dataset."""

    def __init__(
        self, root_dir: str, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for Urban3D based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the Urban3D Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        self.train_dataset = US3D(self.root_dir, split="train", transforms=transforms)
        self.val_dataset = US3D(self.root_dir, split="val", transforms=transforms)
        self.test_dataset = US3D(self.root_dir, split="test", transforms=transforms)

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
