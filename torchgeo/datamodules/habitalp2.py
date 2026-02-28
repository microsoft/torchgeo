# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""HabitAlp2 datamodule."""

from typing import Any, cast

import kornia.augmentation as K
from kornia.constants import DataKey, Resample

from ..datasets import GeoDataset, HabitAlp2, HabitAlp2CD
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..samplers.utils import _to_tuple
from .geo import GeoDataModule


class HabitAlp2DataModule(GeoDataModule):
    """LightningDataModule implementation for the HabitAlp2 dataset.

    Supports both semantic segmentation and change detection tasks.

    .. versionadded:: 0.9.1
    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: int | tuple[int, int] = 256,
        length: int | None = None,
        num_workers: int = 0,
        task: str = 'segmentation',
        **kwargs: Any,
    ) -> None:
        """Initialize a new HabitAlp2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            task: One of 'segmentation' or 'change_detection'.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.HabitAlp2` or
                :class:`~torchgeo.datasets.HabitAlp2CD`.
        """
        assert task in ['segmentation', 'change_detection'], (
            f"task must be 'segmentation' or 'change_detection', got '{task}'"
        )

        self.task = task
        dataset_class = HabitAlp2 if task == 'segmentation' else HabitAlp2CD

        super().__init__(
            dataset_class,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        if task == 'segmentation':
            self.train_aug = K.AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
                K.RandomVerticalFlip(p=0.5),
                K.RandomHorizontalFlip(p=0.5),
                data_keys=None,
                keepdim=True,
                extra_args={
                    DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}
                },
            )
        else:
            self.train_aug = K.AugmentationSequential(
                K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
                data_keys=None,
                keepdim=True,
            )

            self.aug = K.AugmentationSequential(
                K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
                data_keys=None,
                keepdim=True,
            )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = cast(GeoDataset, self.dataset_class(**self.kwargs))
        self.train_dataset = dataset
        self.val_dataset = dataset
        self.test_dataset = dataset

        if stage in ['fit']:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ['fit', 'validate']:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ['test']:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )
