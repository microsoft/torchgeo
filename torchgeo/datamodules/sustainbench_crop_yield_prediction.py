# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Sustainbench Crop Yield Prediction datamodule."""

from typing import Any

from ..datasets import SustainBenchCropYieldPrediction
from .geo import NonGeoDataModule


class SustainbenchCropYieldDataModule(NonGeoDataModule):
    """LightningDataModule for Sustainbench Crop Yield Prediction dataset."""

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SustainbenchCropYieldDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SustainBenchCropYieldPrediction`.

        .. versionadded:: 0.5
        """
        super().__init__(
            SustainBenchCropYieldPrediction, batch_size, num_workers, **kwargs
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="train", **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="dev", **self.kwargs
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="test", **self.kwargs
            )
