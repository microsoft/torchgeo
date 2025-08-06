# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD+ datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import LEVIRCD, LEVIRCDPlus
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _ExtractPatches, _RandomNCrop
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
        super().__init__(
            LEVIRCD, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)

        # Training: Random crops for maximum diversity
        self.train_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                _RandomNCrop(size=self.patch_size, pad_if_needed=True),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=False,  # Allow different crops per batch item
        )
        # Val/Test: Use CenterCrop for compatibility (VideoSequential + _ExtractPatches incompatible)
        # Future improvement: implement proper deterministic patch extraction without VideoSequential
        self.val_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.CenterCrop(size=self.patch_size),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.CenterCrop(size=self.patch_size),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
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
        super().__init__(
            LEVIRCDPlus, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        # Training: Random crops for maximum diversity
        self.train_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                _RandomNCrop(size=self.patch_size, pad_if_needed=True),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=False,  # Allow different crops per batch item
        )
        # Val/Test: Use CenterCrop for compatibility (VideoSequential + _ExtractPatches incompatible)
        # Future improvement: implement proper deterministic patch extraction without VideoSequential
        self.val_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.CenterCrop(size=self.patch_size),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.CenterCrop(size=self.patch_size),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = LEVIRCDPlus(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            self.test_dataset = LEVIRCDPlus(split='test', **self.kwargs)


class LEVIRCDBenchmarkDataModule(NonGeoDataModule):
    """LightningDataModule implementation for LEVIR-CD benchmarking of ChangeViT model.

    This implements from the ChangeViT paper - Each 1024 x 1024 image is divided
    into 16 non-overlapping 256 x 256 patches This results in 7120 pairs for
    training, 1024 pairs for validation, and 2048 pairs for testing."
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
        stride = kwargs.pop('stride', None)

        super().__init__(
            LEVIRCD, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)
        self.stride = _to_tuple(stride) if stride is not None else None

        # Training: Random crops for maximum diversity
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(size=self.patch_size, pad_if_needed=True),
            data_keys=None,  # Dictionary input requires data_keys=None
            keepdim=True,  # Maintain batch structure for temporal data
            same_on_batch=False,  # Allow different crops per batch item
        )

        # For val/test, use same approach as training
        self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(
                window_size=self.patch_size, stride=self.stride, keepdim=False
            ),
            data_keys=None,  # Dictionary input requires data_keys=None
            keepdim=False,  # Allow dimension changes
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(
                window_size=self.patch_size, stride=self.stride, keepdim=False
            ),
            data_keys=None,  # Dictionary input requires data_keys=None
            keepdim=False,  # Allow dimension changes
            same_on_batch=True,
        )

        # Fallback general augmentation (same as val)
        self.aug = self.val_aug

    def setup(self, stage: str) -> None:
        """Set up datasets with transforms.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = LEVIRCD(
                split='train', transforms=self.train_aug, **self.kwargs
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = LEVIRCD(
                split='val', transforms=self.val_aug, **self.kwargs
            )
        if stage in ['test']:
            self.test_dataset = LEVIRCD(
                split='test', transforms=self.test_aug, **self.kwargs
            )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Reshape batch to flatten patches into batch dimension for ChangeViT."""
        # Skip base class transforms to avoid shape issues with patches
        # batch = super().on_after_batch_transfer(batch, dataloader_idx)

        # If patches were extracted, reshape for ChangeViT compatibility
        if len(batch['image'].shape) == 6:  # [B, T, P, C, H, W]
            batch_size, temporal_frames, patches_per_frame = batch['image'].shape[:3]

            # Reshape image: [B, T, P, C, H, W] -> [B*P, T, C, H, W]
            batch['image'] = batch['image'].view(
                batch_size * patches_per_frame,
                temporal_frames,
                *batch['image'].shape[3:],
            )

            # Reshape mask: [B, T, P, C, H, W] -> [B*P, C, H, W]
            if len(batch['mask'].shape) == 6:
                # Permute to [B, P, T, C, H, W] then reshape to [B*P, T*C, H, W]
                batch['mask'] = batch['mask'].permute(
                    0, 2, 1, 3, 4, 5
                )  # [B, P, T, C, H, W]
                batch['mask'] = batch['mask'].reshape(
                    batch_size * patches_per_frame, -1, *batch['mask'].shape[-2:]
                )  # [B*P, T*C, H, W] where T*C = 1*1 = 1

        return batch
