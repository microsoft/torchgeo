# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InriaAerialImageLabeling datamodule."""

from typing import Any, Tuple, Union

import kornia.augmentation as K

from ..datasets import InriaAerialImageLabeling
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.transforms import _ExtractTensorPatches, _RandomNCrop
from .geo import NonGeoDataModule
from .utils import dataset_split


class InriaAerialImageLabelingDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the InriaAerialImageLabeling dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        num_tiles_per_batch: int = 16,
        num_patches_per_tile: int = 16,
        patch_size: Union[Tuple[int, int], int] = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize a new InriaAerialImageLabelingDataModule instance.

        The Inria Aerial Image Labeling dataset contains images that are too large to
        pass directly through a model. Instead, we randomly sample patches from image
        tiles during training and chop up image tiles into patch grids during
        evaluation. During training, the effective batch size is equal to
        ``num_tiles_per_batch`` x ``num_patches_per_tile``.

        Args:
            num_tiles_per_batch: Number of image tiles to sample from during training.
            num_patches_per_tile: Number of patches to randomly sample from each image
                tile during training.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.InriaAerialImageLabeling`.
        """
        super().__init__(InriaAerialImageLabeling, 1, num_workers, **kwargs)

        self.train_batch_size = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            _RandomNCrop(self.patch_size, self.num_patches_per_tile),
            data_keys=["image", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractTensorPatches(self.patch_size),
            data_keys=["image", "mask"],
        )
        self.predict_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractTensorPatches(self.patch_size),
            data_keys=["image"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate", "test"]:
            dataset = InriaAerialImageLabeling(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset, self.val_split_pct, self.test_split_pct
            )
        if stage in ["predict"]:
            # Test set masks are not public, use for prediction instead
            self.predict_dataset = InriaAerialImageLabeling(split="test", **self.kwargs)
