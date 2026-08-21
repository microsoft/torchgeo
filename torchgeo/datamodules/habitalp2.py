# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""HabitAlp2 datamodule."""

import os
from typing import Any, Literal, cast

import geopandas as gpd
import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample

from ..datasets import GeoDataset, HabitAlp2, HabitAlp2CD, random_grid_cell_assignment
from ..datasets.utils import download_url
from ..samplers import GriddedPatchSampler, RandomPatchSampler
from ..samplers.utils import _to_tuple
from .geo import GeoDataModule


class HabitAlp2DataModule(GeoDataModule):
    """LightningDataModule implementation for the HabitAlp2 dataset.

    Supports both semantic segmentation and change detection tasks.

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: int | tuple[int, int] = 256,
        length: int | None = None,
        num_workers: int = 0,
        task: Literal['segmentation', 'change_detection'] = 'segmentation',
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

        Raises:
            AssertionError: if ``task`` is invalid
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
        """Set up datasets and samplers.

        Clips the dataset index to the geographic outline for the target year,
        then applies a random grid cell assignment for train/val/test splitting.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = cast(GeoDataset, self.dataset_class(**self.kwargs))

        # Determine the year for outline selection
        if self.task == 'segmentation':
            year = int(self.kwargs.get('year', '2013'))
        else:
            pair = str(self.kwargs.get('pair', '2013_2020'))
            if '_' not in pair and len(pair) == 8 and pair.isdigit():
                pair = f'{pair[:4]}_{pair[4:]}'
            year = int(pair.split('_')[1])

        # Download outlines GPKG if not present
        outlines_path = os.path.join(dataset.root, HabitAlp2.outlines_file)
        if not os.path.exists(outlines_path):
            os.makedirs(os.path.dirname(outlines_path), exist_ok=True)
            download_url(
                HabitAlp2.url + HabitAlp2.outlines_file,
                dataset.root,
                HabitAlp2.outlines_file,
            )

        # Clip dataset index to the outline for the target year
        outlines = gpd.read_file(outlines_path)
        outline = outlines[outlines['year'] == year].geometry.union_all()
        clipped = dataset.index.copy()
        clipped['geometry'] = clipped.geometry.intersection(outline)
        dataset.index = clipped[~clipped.is_empty]

        # Geographic train/val/test split
        generator = torch.Generator().manual_seed(0)
        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            random_grid_cell_assignment(
                dataset, [0.7, 0.15, 0.15], grid_size=10, generator=generator
            )
        )

        if stage in ['fit']:
            self.train_sampler = RandomPatchSampler(
                self.train_dataset, size=self.patch_size, length=self.length
            )
        if stage in ['fit', 'validate']:
            self.val_sampler = GriddedPatchSampler(
                self.val_dataset, size=self.patch_size, stride=self.patch_size
            )
        if stage in ['test']:
            self.test_sampler = GriddedPatchSampler(
                self.test_dataset, size=self.patch_size, stride=self.patch_size
            )
