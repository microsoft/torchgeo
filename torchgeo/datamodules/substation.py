# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Substation datamodule."""

from typing import Any

import torch
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from ..datasets import Substation
from .geo import NonGeoDataModule


class SubstationDataModule(NonGeoDataModule):
    """Substation Data Module with train-test split and transformations.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        normalizing_type: str = 'percentile',
        normalizing_factor: Any | None = None,
        means: Any | None = None,
        stds: Any | None = None,
        bands: list[int] = [1, 2, 3],
        num_of_timepoints: int = 4,
        model_type: str = 'default',
        geo_transforms: Any | None = None,
        color_transforms: Any | None = None,
        image_resize: Any | None = None,
        mask_resize: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SubstationDataModule instance.

        Args:
            root: Path to the dataset directory.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for data loading.
            val_split_pct: Percentage of data to use for validation.
            test_split_pct: Percentage of data to use for testing.
            normalizing_type: Normalization type ('percentile', 'zscore', or 'default').
            normalizing_factor: Normalization factor for percentile normalization.
            means: Mean values for z-score normalization.
            stds: Standard deviation values for z-score normalization.
            bands: Number of input channels to use.
            model_type: Type of model being used (e.g., 'swin' for specific channel selection).
            geo_transforms: Geometric transformations to apply to the data.
            color_transforms: Color transformations to apply to the image.
            image_resize: Resizing function for the image.
            mask_resize: Resizing function for the mask.
            num_of_timepoints: Number of timepoints to use in the dataset.
            **kwargs: Additional arguments passed to Substation.
        """
        super().__init__(Substation, batch_size, num_workers, **kwargs)
        self.root = root
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.normalizing_type = normalizing_type
        self.normalizing_factor = normalizing_factor
        self.means = means
        self.stds = stds
        self.bands = bands
        self.model_type = model_type
        self.geo_transforms = geo_transforms
        self.color_transforms = color_transforms
        self.image_resize = image_resize
        self.mask_resize = mask_resize
        self.num_of_timepoints = num_of_timepoints

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = Substation(
            root=self.root,
            bands=self.bands,
            use_timepoints=True,
            mask_2d=False,
            num_of_timepoints=self.num_of_timepoints,
            timepoint_aggregation='median',
            download=True,
            checksum=False,
        )

        generator = torch.Generator().manual_seed(0)
        total_len = len(dataset)
        val_len = int(total_len * self.val_split_pct)
        test_len = int(total_len * self.test_split_pct)
        train_len = total_len - val_len - test_len
        print(val_len, test_len, train_len)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_len, val_len, test_len], generator
        )

        if stage == 'fit':
            self.train_dataset = self._apply_transforms(self.train_dataset)
            self.val_dataset = self._apply_transforms(self.val_dataset)
        elif stage == 'test':
            self.test_dataset = self._apply_transforms(self.test_dataset)

    def _apply_transforms(self, dataset: Subset[Any]) -> Subset[Any]:
        """Apply preprocessing and transformations to the dataset.

        Args:
            dataset: A subset of the dataset.

        Returns:
            The processed dataset.
        """
        for sample in tqdm(dataset, desc='Processing images', unit='sample'):
            image, mask = sample['image'], sample['mask']

            if self.geo_transforms:
                combined = torch.cat((image, mask), 0)
                combined = self.geo_transforms(combined)
                image, mask = torch.split(combined, [image.shape[0], mask.shape[0]], 0)

            if self.color_transforms:
                num_timepoints = image.shape[0] // self.bands
                for i in range(num_timepoints):
                    start = i * len(self.bands)
                    end = start + 3
                    image[start:end, :, :] = self.color_transforms(
                        image[start:end, :, :]
                    )

            if self.image_resize:
                image = self.image_resize(image)
            if self.mask_resize:
                mask = self.mask_resize(mask)

            sample['image'], sample['mask'] = image, mask

        return dataset
