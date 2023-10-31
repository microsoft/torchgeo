# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD+ datamodule."""

from typing import Union

import kornia.augmentation as K
import torch

from torchgeo.datamodules.utils import dataset_split
from torchgeo.samplers.utils import _to_tuple

from ..datasets import LEVIRCDPlus
from ..transforms import AugmentationSequential
from ..transforms.transforms import _RandomNCrop
from .geo import NonGeoDataModule


class LEVIRCDPlusDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LEVIR-CD+ dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: Union[tuple[int, int], int] = 256,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs,
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
        super().__init__(LEVIRCDPlusDataModule, 1, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image1", "image2", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = LEVIRCDPlus(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset, _ = dataset_split(
                self.dataset, val_pct=self.val_split_pct, test_pct=0
            )
        if stage in ["test"]:
            self.test_dataset = LEVIRCDPlus(split="test", **self.kwargs)
