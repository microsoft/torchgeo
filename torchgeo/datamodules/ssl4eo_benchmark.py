# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SSL4EO datamodule."""

from typing import Any

import kornia.augmentation as K
from kornia.constants import DataKey, Resample

from ..datasets import SSL4EOLBenchmark
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class SSL4EOLBenchmarkDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-L Benchmark dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 224,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SSL4EOLBenchmarkDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOLBenchmark`.
        """
        super().__init__(SSL4EOLBenchmark, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.patch_size),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.patch_size),
            data_keys=["image", "mask"],
        )
