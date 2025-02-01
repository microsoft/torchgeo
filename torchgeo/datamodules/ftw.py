# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FTW datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import FieldsOfTheWorld
from .geo import NonGeoDataModule


class FieldsOfTheWorldDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FTW dataset.

    .. versionadded:: 0.7
    """

    mean = torch.tensor([0])
    std = torch.tensor([3000])

    def __init__(
        self,
        train_countries: list[str] = ['austria'],
        val_countries: list[str] = ['austria'],
        test_countries: list[str] = ['austria'],
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FTWDataModule instance.

        Args:
            train_countries: List of countries to use for training.
            val_countries: List of countries to use for validation.
            test_countries: List of countries to use for testing.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FieldsOfTheWorld`.

        Raises:
            AssertionError: If 'countries' are specified in kwargs
        """
        assert (
            'countries' not in kwargs
        ), "Please specify 'train_countries', 'val_countries', and 'test_countries' instead of 'countries' inside kwargs"

        super().__init__(FieldsOfTheWorld, batch_size, num_workers, **kwargs)

        self.train_countries = train_countries
        self.val_countries = val_countries
        self.test_countries = test_countries

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            data_keys=None,
            keepdim=True,
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', or 'test'.
        """
        if stage in ['fit', 'validate']:
            self.train_dataset = FieldsOfTheWorld(
                split='train', countries=self.train_countries, **self.kwargs
            )
            self.val_dataset = FieldsOfTheWorld(
                split='val', countries=self.val_countries, **self.kwargs
            )
        if stage in ['test']:
            self.test_dataset = FieldsOfTheWorld(
                split='test', countries=self.test_countries, **self.kwargs
            )
