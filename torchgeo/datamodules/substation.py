# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Substation Data Module."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from ..datasets import Substation


class SubstationDataModule:
    """Substation Data Module with train-test split and transformations.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        split_ratio: float = 0.8,
        normalizing_type: str = 'percentile',
        normalizing_factor: np.ndarray | None = None,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
        bands: int = 13,
        num_of_timepoints: int | None = None,
        model_type: str = 'default',
        geo_transforms: Any = None,
        color_transforms: Any = None,
        image_resize: Any = None,
        mask_resize: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SubstationDataModule instance.

        Args:
            root: Path to the dataset directory.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for data loading.
            split_ratio: Ratio of data to use for training.
            normalizing_type: Normalization type ('percentile', 'zscore', or 'default').
            normalizing_factor: Normalization factor for percentile normalization.
            means: Mean values for z-score normalization.
            stds: Standard deviation values for z-score normalization.
            num_of_timepoints: Number of timepoints to use.
            bands: Number of input channels to use.
            model_type: Type of model being used (e.g., 'swin' for specific channel selection).
            geo_transforms: Geometric transformations to apply to the data.
            color_transforms: Color transformations to apply to the image.
            image_resize: Resizing function for the image.
            mask_resize: Resizing function for the mask.
            **kwargs: Additional arguments passed to Substation.
        """
        # super().__init__(
        #     Substation,
        #     root=root,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     **kwargs,
        # )
        self.root = root
        self.split_ratio = split_ratio
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

        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
            timepoint_aggregation='concat',
            download=True,
            checksum=False,
        )

        # Train-test split
        total_size = len(dataset)
        train_size = int(total_size * self.split_ratio)
        indices = list(range(total_size))
        train_indices, test_indices = torch.utils.data.random_split(
            indices, [train_size, total_size - train_size]
        )

        if stage in ['fit', 'validate']:
            # Further split train set into train imageand validation sets
            val_split_ratio = 0.2
            val_size = int(len(train_indices) * val_split_ratio)
            train_size = len(train_indices) - val_size
            train_indices, val_indices = torch.utils.data.random_split(
                train_indices, [train_size, val_size]
            )

            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)

            # Apply preprocessing to train and validation datasets
            self.train_dataset = self._apply_transforms(self.train_dataset)
            self.val_dataset = self._apply_transforms(self.val_dataset)

        if stage == 'test':
            self.test_dataset = Subset(dataset, test_indices)
            self.test_dataset = self._apply_transforms(self.test_dataset)

    def _apply_transforms(self, dataset: Subset) -> Subset:
        """Apply preprocessing and transformations to the dataset.

        Args:
            dataset: A subset of the dataset.

        Returns:
            The processed dataset.
        """
        for sample in tqdm(dataset, desc='Processing images', unit='sample'):
            image, mask = sample['image'], sample['mask']

            # Standardizing image
            # if self.normalizing_type == "percentile":
            #     image = (image - self.normalizing_factor[:, 0].reshape((-1, 1, 1))) / self.normalizing_factor[:, 2].reshape((-1, 1, 1))
            # elif self.normalizing_type == "zscore":
            #     image = (image - self.means) / self.stds
            # else:
            #     image = image / self.normalizing_factor
            #     image = torch.clamp(image, 0, 1)

            # Applying geometric transformations
            if self.geo_transforms:
                combined = torch.cat((image, mask), 0)
                combined = self.geo_transforms(combined)
                image, mask = torch.split(combined, [image.shape[0], mask.shape[0]], 0)

            # Applying color transformations
            if self.color_transforms:
                num_timepoints = image.shape[0] // self.bands
                for i in range(num_timepoints):
                    if self.bands >= 3:
                        start = i * self.bands
                        end = start + 3
                        image[start:end, :, :] = self.color_transforms(
                            image[start:end, :, :]
                        )
                    else:
                        raise ValueError(
                            'Input dimensions must support color transformations.'
                        )

            # Resizing image and mask
            if self.image_resize:
                image = self.image_resize(image)
            if self.mask_resize:
                mask = self.mask_resize(mask)

            sample['image'], sample['mask'] = image, mask

        return dataset
