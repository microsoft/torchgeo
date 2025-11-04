# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD datamodules."""

from typing import Any, Literal, cast

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
    """Unified LightningDataModule for LEVIR-CD and LEVIR-CD+ datasets.

    Implements the modern random training + deterministic evaluation strategy
    with proper temporal correspondence fixes.

    .. versionadded:: 0.6
    .. versionchanged:: 0.8
        Unified interface supporting both LEVIR-CD and LEVIR-CD+ datasets.
        Implements modern patch extraction methodology.
    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: tuple[int, int] | int = 256,
        dataset_variant: Literal['levircd', 'levircd-plus'] = 'levircd',
        val_split_pct: float | None = None,
        num_workers: int = 0,
        val_patch_sampling: Literal['grid', 'random'] = 'random',
        **kwargs: Any,
    ) -> None:
        """Initialize a new LEVIRCDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            dataset_variant: Which dataset variant to use ('levircd' or 'levircd-plus').
            val_split_pct: Percentage of the dataset to use as a validation set.
                Only used for 'levircd-plus' variant. Defaults to None for 'levircd'
                (uses official splits) and 0.2 for 'levircd-plus'.
            num_workers: Number of workers for parallel data loading.
            val_patch_sampling: Strategy for validation patch sampling:
                'random' (default): Use RandomCrop, matches training distribution.
                'grid': Use deterministic grid patches via ExtractPatches.
            **kwargs: Additional keyword arguments passed to the dataset.
        """
        # Handle stride parameter for patch extraction
        stride = kwargs.pop('stride', None)
        self.val_patch_sampling = val_patch_sampling

        # Select dataset class based on variant
        dataset_class: type[LEVIRCD | LEVIRCDPlus]
        if dataset_variant == 'levircd':
            dataset_class = cast(type[LEVIRCD | LEVIRCDPlus], LEVIRCD)
            # Default to None for official splits
            if val_split_pct is None:
                val_split_pct = None
        elif dataset_variant == 'levircd-plus':
            dataset_class = cast(type[LEVIRCD | LEVIRCDPlus], LEVIRCDPlus)
            # Default to 0.2 for LEVIR-CD+ random splits
            if val_split_pct is None:
                val_split_pct = 0.2
        else:
            raise ValueError(
                f"dataset_variant must be 'levircd' or 'levircd-plus', got {dataset_variant}"
            )

        super().__init__(
            dataset_class, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)
        self.stride = _to_tuple(stride) if stride is not None else None
        self.dataset_variant = dataset_variant
        self.val_split_pct = val_split_pct

        # Create transforms using factory methods
        self.train_aug = self._create_random_train_aug()
        self.val_aug = self._create_deterministic_val_aug()
        self.test_aug = self._create_deterministic_val_aug()  # Same as val
        self.aug = self.val_aug  # Fallback general augmentation

    def _create_random_train_aug(self) -> K.AugmentationSequential:
        """Create synchronized random training augmentation for proper image-mask alignment.

        Follows ChangeViT paper: random flipping and cropping.
        """
        return K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )

    def _create_deterministic_val_aug(self) -> K.AugmentationSequential:
        """Create validation/test augmentation based on val_patch_sampling strategy."""
        if self.val_patch_sampling == 'random':
            # Use RandomCrop to match training distribution
            return K.AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.RandomCrop(self.patch_size, pad_if_needed=True),
                data_keys=None,
                keepdim=True,
                same_on_batch=True,
            )
        else:  # 'grid'
            # Use deterministic grid patches for comprehensive evaluation
            return K.AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                _ExtractPatches(
                    window_size=self.patch_size, stride=self.stride, keepdim=False
                ),
                data_keys=None,
                keepdim=False,  # Allow dimension changes for patches
                same_on_batch=True,
            )

    def setup(self, stage: str) -> None:
        """Set up datasets with transforms.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self.dataset_variant == 'levircd':
            # LEVIR-CD: Use official train/val/test splits
            if stage in ['fit']:
                self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                    split='train', transforms=self.train_aug, **self.kwargs
                )
            if stage in ['fit', 'validate']:
                self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                    split='val', transforms=self.val_aug, **self.kwargs
                )
            if stage in ['test']:
                self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                    split='test', transforms=self.test_aug, **self.kwargs
                )

        elif self.dataset_variant == 'levircd-plus':
            # LEVIR-CD+: Create train/val split from train set, separate test set
            if stage in ['fit', 'validate']:
                full_dataset = self.dataset_class(split='train', **self.kwargs)  # type: ignore[call-arg]
                generator = torch.Generator().manual_seed(0)
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset,
                    [1 - self.val_split_pct, self.val_split_pct],  # type: ignore[operator,list-item]
                    generator,
                )
                # Apply transforms after splitting
                self.train_dataset.dataset.transforms = self.train_aug  # type: ignore[attr-defined]
                self.val_dataset.dataset.transforms = self.val_aug  # type: ignore[attr-defined]
            if stage in ['test']:
                self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                    split='test', transforms=self.test_aug, **self.kwargs
                )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Reshape batch to flatten patches into batch dimension for ChangeViT compatibility."""
        # If patches were extracted (grid mode), reshape for ChangeViT compatibility
        # In random crop mode, batch is already [B, T, C, H, W] so no reshaping needed
        if len(batch['image'].shape) == 6:  # [B, T, P, C, H, W]
            # Reshape image: [B, T, P, C, H, W] -> [B*P, T, C, H, W]
            batch['image'] = rearrange(batch['image'], 'b t p c h w -> (b p) t c h w')

            # Reshape mask: [B, T, P, C, H, W] -> [B*P, C, H, W]
            if len(batch['mask'].shape) == 6:
                # Squeeze temporal dimension (T=1) then rearrange patches into batch
                batch['mask'] = batch['mask'].squeeze(1)  # [B, P, C, H, W]
                batch['mask'] = rearrange(batch['mask'], 'b p c h w -> (b p) c h w')

        # Handle mask reshaping for cases where temporal dimension creates extra dimensions
        if len(batch['mask'].shape) == 5 and batch['mask'].shape[2] == 1:
            # Handle case where mask is [B, C, 1, H, W] -> [B, C, H, W] (remove temporal dim)
            batch['mask'] = batch['mask'].squeeze(2)

        return batch


# Backward compatibility alias
LEVIRCDPlusDataModule = LEVIRCDDataModule
