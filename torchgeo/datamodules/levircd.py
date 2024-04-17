# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD+ datamodule."""

from typing import Any

import kornia.augmentation as K
from torch.utils.data import random_split

from torchgeo.samplers.utils import _to_tuple

from ..datasets import LEVIRCD, LEVIRCDPlus
from ..transforms import AugmentationSequential
from ..transforms.transforms import _RandomNCrop
from .geo import NonGeoDataModule


class LEVIRCDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LEVIR-CD dataset.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: tuple[int, int] | int = 256,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LEVIRCDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LEVIRCD`.
        """
        super().__init__(LEVIRCD, 1, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image1", "image2", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image1", "image2", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image1", "image2", "mask"],
        )


class LEVIRCDPlusDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LEVIR-CD+ dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: tuple[int, int] | int = 256,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LEVIRCDPlusDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LEVIRCDPlus`.
        """
        super().__init__(LEVIRCDPlus, 1, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image1", "image2", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image1", "image2", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image1", "image2", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = LEVIRCDPlus(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, val_pct=self.val_split_pct
            )
        if stage in ["test"]:
            self.test_dataset = LEVIRCDPlus(split="test", **self.kwargs)
