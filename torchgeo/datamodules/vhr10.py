# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NWPU VHR-10 datamodule."""

from typing import Any, Dict, List

import torch
import torchvision
from torch import Tensor

from ..datasets import VHR10
from ..samplers.utils import _to_tuple
from .geo import NonGeoDataModule
from .utils import dataset_split


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Any]:
    """Custom object detection collate fn to handle variable boxes.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output
    """
    output: Dict[str, Any] = {}
    output["image"] = torch.stack([sample["image"] for sample in batch])
    output["boxes"] = [sample["boxes"] for sample in batch]
    output["labels"] = [sample["labels"] for sample in batch]
    if "masks" in batch[0]:
        output["masks"] = [sample["masks"] for sample in batch]
    return output


class VHR10DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the VHR10 dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int = 512,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new VHR10DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.VHR10`.
        """
        super().__init__(VHR10, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = _to_tuple(patch_size)

        self.collate_fn = collate_fn
        self.kwargs["transforms"] = self.preprocess
        self.kwargs["download"] = True

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()

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

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = VHR10(**self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )
