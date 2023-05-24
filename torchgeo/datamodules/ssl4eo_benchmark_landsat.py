# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SSL4EO datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import SSL4EOLBenchmark
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class SSL4EOLBenchmarkDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-L dataset.

    .. versionadded:: 0.5
    """

    mean = 0
    std = 255

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOLDownstreamDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOLBenchmark`.
        """
        super().__init__(SSL4EOLBenchmark, batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(224),
            data_keys=["image", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(224),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(224),
            data_keys=["image", "mask"],
        )
        self.predict_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(224),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SSL4EOLBenchmark(**self.kwargs)
