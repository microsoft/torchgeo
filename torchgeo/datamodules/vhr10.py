# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NWPU VHR-10 datamodule."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import VHR10
from ..samplers.utils import _to_tuple
from .utils import dataset_split

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Any]:
    """Custom object detection collate fn to handle variable boxes.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output
    """
    output: Dict[str, Any] = {}
    output["image"] = torch.stack([sample["image"] for sample in batch])
    output["labels"] = [sample["labels"] for sample in batch]
    output["boxes"] = [sample["boxes"] for sample in batch]
    if "masks" in batch[0]:
        output["masks"] = [sample["masks"] for sample in batch]
    return output


class VHR10DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the VHR10 dataset.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        patch_size: Union[int, Tuple[int, int]] = 512,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for VHR10 based DataLoaders.

        Args:
            root: The ``root`` argument to pass to the Dataset class
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            patch_size: Patch size (height, width) for batched training
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = cast(Tuple[int, int], _to_tuple(patch_size))

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0

        _, h, w = sample["image"].shape
        sample["image"] = torchvision.transforms.functional.resize(
            sample["image"], size=self.patch_size
        )
        box_scale = (self.patch_size[1] / w, self.patch_size[0] / h)
        sample["boxes"][:, 0] *= box_scale[0]
        sample["boxes"][:, 1] *= box_scale[1]
        sample["boxes"][:, 2] *= box_scale[0]
        sample["boxes"][:, 3] *= box_scale[1]
        sample["boxes"] = torch.round(sample["boxes"])

        if "masks" in sample:
            sample["masks"] = torchvision.transforms.functional.resize(
                sample["masks"], size=self.patch_size
            )

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        VHR10(self.root, download=True, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.dataset = VHR10(self.root, transforms=self.preprocess)
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

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.VHR10.plot`."""
        return self.dataset.plot(*args, **kwargs)
