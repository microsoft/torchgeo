# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DeepGlobe Land Cover Classification Challenge datamodule."""

from typing import Any, Dict, Optional, Tuple, Union

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from ..datasets import DeepGlobeLandCover
from ..transforms import AugmentationSequential, NCrop, PadToMultiple
from .utils import dataset_split


class DeepGlobeLandCoverDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the DeepGlobe Land Cover dataset.

    Uses the train/test splits from the dataset.
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
        """Initialize a new LightningDataModule instance.

        The DeepGlobe Land Cover dataset contains images that are too large to pass
        directly through a model. Instead, we randomly sample patches from image tiles
        during training and chop up image tiles into patch grids during evaluation.
        During training, the effective batch size is equal to
        ``num_tiles_per_batch`` x ``num_patches_per_tile``.

        Args:
            num_tiles_per_batch: The number of image tiles to sample from during
                training
            num_patches_per_tile: The number of patches to randomly sample from each
                image tile during training
            patch_size: The size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures
            val_split_pct: The percentage of the dataset to use as a validation set
            num_workers: The number of workers to use for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DeepGlobeLandCover`

        .. versionchanged:: 0.4
           *batch_size* was replaced by *num_tile_per_batch*, *num_patches_per_tile*,
           and *patch_size*.
        """
        super().__init__()

        self.num_tiles_per_batch = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        self.patch_size = patch_size
        self.val_split_pct = val_split_pct
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.train_transform = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0),
            NCrop(patch_size, self.num_patches_per_tile),
            data_keys=["image", "mask"],
        )
        self.test_transform = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0),
            PadToMultiple(32),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: the stage to set up
        """
        train_dataset = DeepGlobeLandCover(
            split="train", transforms=self.train_transform, **self.kwargs
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            val_dataset = DeepGlobeLandCover(
                split="train", transforms=self.test_transform, **self.kwargs
            )
            self.train_dataset, self.val_dataset, _ = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
            self.val_dataset.dataset = val_dataset
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = DeepGlobeLandCover(
            split="test", transforms=self.test_transform, **self.kwargs
        )

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.num_tiles_per_batch,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.DeepGlobeLandCover.plot`.

        .. versionadded:: 0.4
        """
        return self.test_dataset.plot(*args, **kwargs)
