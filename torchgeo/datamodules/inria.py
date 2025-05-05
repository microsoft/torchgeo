# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InriaAerialImageLabeling datamodule."""

from typing import Any

import kornia.augmentation as K
from torch import Tensor

from ..datasets import InriaAerialImageLabeling
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _ExtractPatches
from .geo import NonGeoDataModule


class InriaAerialImageLabelingDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the InriaAerialImageLabeling dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new InriaAerialImageLabelingDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.InriaAerialImageLabeling`.
        """
        super().__init__(
            InriaAerialImageLabeling,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.patch_size = _to_tuple(patch_size)

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            data_keys=None,
            keepdim=True,
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(window_size=self.patch_size),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.predict_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(window_size=self.patch_size),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = InriaAerialImageLabeling(split='train', **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = InriaAerialImageLabeling(split='val', **self.kwargs)
        if stage in ['predict']:
            # Test set masks are not public, use for prediction instead
            self.predict_dataset = InriaAerialImageLabeling(split='test', **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.

        .. versionadded:: 0.7
        """
        # This solves a special case where if batch_size=1 the mask won't be stacked correctly
        if 'mask' in batch and batch['mask'].ndim == 3:
            batch['mask'] = batch['mask'].unsqueeze(0)
            batch = super().on_after_batch_transfer(batch, dataloader_idx)
            batch['mask'] = batch['mask'].squeeze(dim=1)
            return batch
        else:
            return super().on_after_batch_transfer(batch, dataloader_idx)
