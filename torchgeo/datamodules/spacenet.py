# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datamodules."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import SpaceNet, SpaceNet1, SpaceNet6
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class SpaceNetBaseDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SpaceNet datasets.

    Randomly splits the train split into train/val/test. The test split does not have labels,
    and is only used for prediction.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        spacenet_ds_class: type[SpaceNet],
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SpaceNetBaseDataModule instance.

        Args:
            spacenet_ds_class: The SpaceNet dataset class to use.
            batch_size: Size of each mini-batch.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to the SpaceNet dataset.
        """
        super().__init__(spacenet_ds_class, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.spacenet_ds_class = spacenet_ds_class

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate', 'test']:
            self.dataset = self.spacenet_ds_class(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [
                    1 - self.val_split_pct - self.test_split_pct,
                    self.val_split_pct,
                    self.test_split_pct,
                ],
                generator,
            )

        # test split in SpaceNet does not have labels
        if stage in ['predict']:
            self.predict_dataset = self.spacenet_ds_class(split='test', **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        if 'mask' in batch:
            batch['mask'] += 1

        return super().on_after_batch_transfer(batch, dataloader_idx)


class SpaceNet1DataModule(SpaceNetBaseDataModule):
    """LightningDataModule implementation for the SpaceNet1 dataset.

    Randomly splits into train/val/test.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SpaceNet1DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SpaceNet1`.
        """
        super().__init__(
            SpaceNet1, batch_size, num_workers, val_split_pct, test_split_pct, **kwargs
        )

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.PadTo((448, 448)),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=['image', 'mask'],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.PadTo((448, 448)),
            data_keys=['image', 'mask'],
        )


class SpaceNet6DataModule(SpaceNetBaseDataModule):
    """LightningDataModule implementation for the SpaceNet6 dataset.

    Randomly splits the training set into train/val and uses the designated test set.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SpaceNet6DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SpaceNet6`.
        """
        super().__init__(
            SpaceNet6, batch_size, num_workers, val_split_pct, test_split_pct, **kwargs
        )
