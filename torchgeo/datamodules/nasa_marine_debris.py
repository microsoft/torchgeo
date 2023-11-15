# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Marine Debris datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor

from ..datasets import NASAMarineDebris
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule
from .utils import AugPipe, dataset_split


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Any]:
    """Custom object detection collate fn to handle variable boxes.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output
    """
    output: dict[str, Any] = {}
    output["image"] = [sample["image"] for sample in batch]
    output["boxes"] = [sample["boxes"].float() for sample in batch]
    output["labels"] = [torch.tensor([1] * len(sample["boxes"])) for sample in batch]
    return output


class NASAMarineDebrisDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the NASA Marine Debris dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new NASAMarineDebrisDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NASAMarineDebris`.
        """
        super().__init__(NASAMarineDebris, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.aug = AugPipe(
            AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "boxes"]
            ),
            batch_size,
        )

        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = NASAMarineDebris(**self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )
