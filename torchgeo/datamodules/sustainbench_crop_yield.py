# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SustainBench Crop Yield datamodule."""

from typing import Any

from ..datasets import SustainBenchCropYield
from .geo import NonGeoDataModule


class SustainBenchCropYieldDataModule(NonGeoDataModule):
    """LightningDataModule for SustainBench Crop Yield dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SustainBenchCropYieldDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SustainBenchCropYield`.
        """
        super().__init__(SustainBenchCropYield, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = SustainBenchCropYield(split="train", **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = SustainBenchCropYield(split="dev", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = SustainBenchCropYield(split="test", **self.kwargs)
