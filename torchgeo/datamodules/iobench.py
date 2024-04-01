# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""I/O benchmark datamodule."""

from typing import Any, Optional, Union

from ..datasets import IOBench
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from .geo import GeoDataModule


class IOBenchDataModule(GeoDataModule):
    """LightningDataModule implementation for the I/O benchmark dataset.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 32,
        patch_size: Union[int, tuple[int, int]] = 512,
        length: Optional[int] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new IOBenchDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.IOBench`.
        """
        super().__init__(
            IOBench,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = IOBench(**self.kwargs)

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )
