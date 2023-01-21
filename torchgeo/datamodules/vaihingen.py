# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Vaihingen datamodule."""

from typing import Any, Tuple, Union

import kornia.augmentation as K

from ..datasets import Vaihingen2D
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.transforms import _RandomNCrop
from .geo import NonGeoDataModule
from .utils import dataset_split


class Vaihingen2DDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Vaihingen2D dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        num_tiles_per_batch: int = 16,
        num_patches_per_tile: int = 16,
        patch_size: Union[Tuple[int, int], int] = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Vaihingen2DDataModule instance.

        The Vaihingen2D dataset contains images that are too large to pass
        directly through a model. Instead, we randomly sample patches from image tiles.
        The effective batch size is equal to
        ``num_tiles_per_batch`` x ``num_patches_per_tile``.

        .. versionchanged:: 0.4
           *batch_size* was replaced by *num_tile_per_batch*, *num_patches_per_tile*,
           and *patch_size*.

        Args:
            num_tiles_per_batch: Number of image tiles to sample from.
            num_patches_per_tile: Number of patches to randomly sample from each image
                tile.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Vaihingen2D`.
        """
        super().__init__(Vaihingen2D, 1, num_workers, **kwargs)

        self.train_batch_size = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, self.num_patches_per_tile),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = Vaihingen2D(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset = dataset_split(
                self.dataset, self.val_split_pct
            )
        if stage in ["test"]:
            self.test_dataset = Vaihingen2D(split="test", **self.kwargs)
