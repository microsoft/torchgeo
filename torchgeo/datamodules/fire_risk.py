# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FireRisk datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import FireRisk
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class FireRiskDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    .. versionadded:: 0.5
    """

    mean = torch.tensor([0, 0, 0])
    std = torch.tensor([255.0, 255.0, 255.0])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new FireRiskDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FireRisk`.
        """
        super().__init__(FireRisk, batch_size, num_workers, **kwargs)
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="train", **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="val", **self.kwargs
            )
        if stage in ["test"]:
            # FireRisk has no test set
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="val", **self.kwargs
            )
