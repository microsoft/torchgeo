# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD datamodules."""

from typing import Any

import kornia.augmentation as K
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import LEVIRCD, LEVIRCDPlus
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _ExtractPatches
from .geo import NonGeoDataModule


class LEVIRCDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LEVIR-CD dataset.

    Implements modern random training + deterministic evaluation strategy
    with proper temporal correspondence fixes.

    .. versionadded:: 0.6
    .. versionchanged:: 0.8
        Fixed temporal correspondence issues from GitHub issue #2920.
        Added val_patch_sampling parameter for flexible evaluation strategies.
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
        super().__init__(
            LEVIRCD, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)

        # Training: random augmentation
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )

        # Validation and test: deterministic grid sampling
        self.val_aug = self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(window_size=self.patch_size, keepdim=False),
            data_keys=None,
            keepdim=False,
            same_on_batch=True,
        )

        self.aug = self.val_aug

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Reshape batch to flatten patches into batch dimension for ChangeViT compatibility."""
        if len(batch['image'].shape) == 6:
            batch['image'] = rearrange(batch['image'], 'b t p c h w -> (b p) t c h w')

            if len(batch['mask'].shape) == 6:
                batch['mask'] = batch['mask'].squeeze(1)
                batch['mask'] = rearrange(batch['mask'], 'b p c h w -> (b p) c h w')

        if len(batch['mask'].shape) == 5 and batch['mask'].shape[2] == 1:
            batch['mask'] = batch['mask'].squeeze(2)

        return batch


class LEVIRCDPlusDataModule(LEVIRCDDataModule):
    """LightningDataModule implementation for the LEVIR-CD+ dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.6
    .. versionchanged:: 0.8
        Fixed temporal correspondence issues from GitHub issue #2920.
        Added val_patch_sampling parameter for flexible evaluation strategies.
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
        self.val_split_pct = val_split_pct

        NonGeoDataModule.__init__(
            self, LEVIRCDPlus, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)

        # Training: random augmentation
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )

        # Validation and test: deterministic grid sampling
        self.val_aug = self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(window_size=self.patch_size, keepdim=False),
            data_keys=None,
            keepdim=False,
            same_on_batch=True,
        )

        self.aug = self.val_aug

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            full_dataset = LEVIRCDPlus(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
            self.train_dataset.dataset.transforms = self.train_aug  # type: ignore[attr-defined]
            self.val_dataset.dataset.transforms = self.val_aug  # type: ignore[attr-defined]
        if stage in ['test']:
            self.test_dataset = LEVIRCDPlus(split='test', **self.kwargs)
