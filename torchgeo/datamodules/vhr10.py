# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NWPU VHR-10 datamodule."""

from typing import Any, Callable, Union

import kornia.augmentation as K
from einops import rearrange
from torch import Tensor
from torch.nn import Module

from ..datasets import VHR10
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule
from .utils import dataset_split


class _AugPipe(Module):
    """Pipeline for applying augmentations sequentially on select data keys."""

    def __init__(
        self, augs: Callable[[dict[str, Any]], dict[str, Any]], batch_size: int
    ) -> None:
        """Initialize a new _AugPipe instance.

        Args:
            augs: Augmentations to apply.
            batch_size: Batch size
        """
        super().__init__()
        self.augs = augs
        self.batch_size = batch_size

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply the augmentation.

        Args:
            batch: Input batch.

        Returns:
            Augmented batch.
        """
        batch_len = len(batch["image"])
        for bs in range(batch_len):
            batch_dict = {
                "image": batch["image"][bs],
                "masks": batch["masks"][bs],
                "labels": batch["labels"][bs],
                "boxes": batch["boxes"][bs],
            }

            batch_dict = self.augs(batch_dict)

            batch["image"][bs] = batch_dict["image"]
            batch["masks"][bs] = batch_dict["masks"]
            batch["labels"][bs] = batch_dict["labels"]
            batch["boxes"][bs] = batch_dict["boxes"]

        # Stack images
        batch["image"] = rearrange(batch["image"], "b () c h w -> b c h w")

        return batch


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Any]:
    """Custom object detection collate fn to handle variable boxes.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output
    """
    output: dict[str, Any] = {}
    output["image"] = [sample["image"] for sample in batch]
    output["boxes"] = [sample["boxes"] for sample in batch]
    output["labels"] = [sample["labels"] for sample in batch]
    output["masks"] = [sample["masks"] for sample in batch]
    return output


class VHR10DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the VHR10 dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[tuple[int, int], int] = 512,
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

        self.train_aug = _AugPipe(
            AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(self.patch_size),
                K.RandomHorizontalFlip(),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.7),
                K.RandomVerticalFlip(),
                data_keys=["image", "boxes", "masks"],
            ),
            batch_size,
        )
        self.aug = _AugPipe(
            AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(self.patch_size),
                data_keys=["image", "boxes", "masks"],
            ),
            batch_size,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = VHR10(**self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, self.val_split_pct, self.test_split_pct
        )
