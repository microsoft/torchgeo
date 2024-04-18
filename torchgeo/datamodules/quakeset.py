# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""QuakeSet datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import QuakeSet
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class QuakeSetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the QuakeSet dataset.

    .. versionadded:: 0.6
    """

    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new QuakeSetDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.QuakeSet`.
        """
        super().__init__(QuakeSet, batch_size, num_workers, **kwargs)
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["image"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = QuakeSet(split="train", **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = QuakeSet(split="validation", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = QuakeSet(split="test", **self.kwargs)
