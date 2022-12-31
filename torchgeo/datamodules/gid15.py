# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""GID-15 datamodule."""

from typing import Any, Optional, Tuple, Union

from kornia.augmentation import Normalize

from ..datasets import GID15
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.transforms import _ExtractTensorPatches, _RandomNCrop
from .geo import NonGeoDataModule
from .utils import dataset_split


class GID15DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the GID-15 dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        num_tiles_per_batch: int = 16,
        num_patches_per_tile: int = 16,
        patch_size: Union[Tuple[int, int], int] = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LightningDataModule instance.

        The GID-15 dataset contains images that are too large to pass
        directly through a model. Instead, we randomly sample patches from image tiles
        during training and chop up image tiles into patch grids during evaluation.
        During training, the effective batch size is equal to
        ``num_tiles_per_batch`` x ``num_patches_per_tile``.

        Args:
            num_tiles_per_batch: The number of image tiles to sample from during
                training
            num_patches_per_tile: The number of patches to randomly sample from each
                image tile during training
            patch_size: The size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures
            val_split_pct: The percentage of the dataset to use as a validation set
            num_workers: The number of workers to use for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.GID15`
        """
        super().__init__()

        self.train_batch_size = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.train_aug = AugmentationSequential(
            Normalize(mean=0.0, std=255.0),
            _RandomNCrop(self.patch_size, self.num_patches_per_tile),
            data_keys=["image", "mask"],
        )
        self.val_aug = AugmentationSequential(
            Normalize(mean=0.0, std=255.0),
            _ExtractTensorPatches(self.patch_size),
            data_keys=["image", "mask"],
        )
        self.predict_transform = AugmentationSequential(
            Normalize(mean=0.0, std=255.0),
            _ExtractTensorPatches(self.patch_size),
            data_keys=["image"],
        )

    def prepare_data(self) -> None:
        """Initialize the main Dataset objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        if self.kwargs.get("download", False):
            GID15(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_dataset = GID15(split="train", **self.kwargs)
        self.train_dataset, self.val_dataset = dataset_split(
            train_dataset, self.val_split_pct
        )

        # Test set masks are not public, use for prediction instead
        self.predict_dataset = GID15(split="test", **self.kwargs)
