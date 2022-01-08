# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Urban3DChallenge datamodule."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import Urban3DChallenge

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class Urban3DChallengeDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the Urban3DChallenge dataset.

    .. versionadded:: 0.3
    """

    # global min/max values of train set
    band_mins = torch.tensor(  # type: ignore[attr-defined]
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0]
    )
    band_maxs = torch.tensor(  # type: ignore[attr-defined]
        [6.0, 16.0, 9859.0, 12872.0, 13163.0, 14445.0]
    )

    def __init__(
        self, root_dir: str, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for Urban3DChallenge based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the Urban3D Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mins = self.band_mins[:, None, None]
        self.maxs = self.band_maxs[:, None, None]

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = (sample["image"] - self.mins) / (self.maxs - self.mins)
        sample["image"] = torch.clamp(  # type: ignore[attr-defined]
            sample["image"], min=0.0, max=1.0
        )
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        self.train_dataset = Urban3DChallenge(
            self.root_dir, split="train", transforms=transforms
        )
        self.val_dataset = Urban3DChallenge(
            self.root_dir, split="val", transforms=transforms
        )
        self.test_dataset = Urban3DChallenge(
            self.root_dir, split="test", transforms=transforms
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
