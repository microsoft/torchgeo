# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LEVIR-CD+ datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import LEVIRCD, LEVIRCDPlus
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _ExtractPatches
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
        model: str = '',
        **kwargs: Any,
    ) -> None:
        """Initialize a new LEVIRCDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            num_workers: Number of workers for parallel data loading.
            model: Model name (used to adjust augmentations for specific models).
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LEVIRCD`.
        """
        super().__init__(
            LEVIRCD, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)
        self.model = model

        self.train_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.RandomCrop(self.patch_size, pad_if_needed=True),
            ),
            data_keys=None,
            keepdim=True,
        )

        # Use CenterCrop for ChangeViT models to avoid multiple patches
        if model.startswith('changevit'):
            self.val_aug = K.AugmentationSequential(
                K.VideoSequential(
                    K.Normalize(mean=self.mean, std=self.std),
                    K.CenterCrop(self.patch_size),
                ),
                data_keys=None,
                keepdim=True,
            )
            self.test_aug = K.AugmentationSequential(
                K.VideoSequential(
                    K.Normalize(mean=self.mean, std=self.std),
                    K.CenterCrop(self.patch_size),
                ),
                data_keys=None,
                keepdim=True,
            )
        else:
            self.val_aug = K.AugmentationSequential(
                K.VideoSequential(
                    K.Normalize(mean=self.mean, std=self.std),
                    _ExtractPatches(window_size=self.patch_size),
                ),
                data_keys=None,
                keepdim=True,
                same_on_batch=True,
            )
            self.test_aug = K.AugmentationSequential(
                K.VideoSequential(
                    K.Normalize(mean=self.mean, std=self.std),
                    _ExtractPatches(window_size=self.patch_size),
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

        self.train_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.RandomCrop(self.patch_size, pad_if_needed=True),
            ),
            data_keys=None,
            keepdim=True,
        )
        self.val_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                _ExtractPatches(window_size=self.patch_size),
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std),
                _ExtractPatches(window_size=self.patch_size),
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
