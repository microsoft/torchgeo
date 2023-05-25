# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SSL4EO datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import SSL4EOLBenchmark
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class SSL4EOLBenchmarkDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-L Benchmark dataset.

    .. versionadded:: 0.5
    """

    crop_size = 224

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOLBenchmarkDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOLBenchmark`.
        """
        super().__init__(SSL4EOLBenchmark, batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.crop_size),
            data_keys=["image", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.crop_size),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.crop_size),
            data_keys=["image", "mask"],
        )
