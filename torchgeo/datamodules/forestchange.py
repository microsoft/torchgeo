# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Forest Change datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import ForestChange
from ..samplers.utils import _to_tuple
from .geo import NonGeoDataModule


class ForestChangeDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Forest Change dataset.

    .. versionadded:: 0.10
    """

    # Calculates dataset-wide per-channel mean and standard deviation by aggregating pixel-wise sums
    #  and squared sums across all images using a numerically stable variance formulation.
    mean = (0.2267 * 255, 0.29982 * 255, 0.22058 * 255)
    std = (0.0923 * 255, 0.06658 * 255, 0.05681 * 255)

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: tuple[int, int] | int = 256,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialise a new ForestChangeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ForestChange`.
        """
        super().__init__(
            ForestChange, batch_size=batch_size, num_workers=num_workers, **kwargs
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
                K.Normalize(mean=self.mean, std=self.std), K.CenterCrop(self.patch_size)
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.VideoSequential(
                K.Normalize(mean=self.mean, std=self.std), K.CenterCrop(self.patch_size)
            ),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.aug = self.train_aug

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = ForestChange(split="train", **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = ForestChange(split="val", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = ForestChange(split="test", **self.kwargs)
