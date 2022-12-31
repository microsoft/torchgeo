# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LoveDA datamodule."""

from typing import Any, Optional

from kornia.augmentation import Normalize

from ..datasets import LoveDA
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class LoveDADataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LoveDA dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for LoveDA based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LoveDA`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            Normalize(mean=0.0 std=255.0), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            LoveDA(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = LoveDA(split="train", **self.kwargs)
        self.val_dataset = LoveDA(split="val", **self.kwargs)

        # Test set masks are not public, use for prediction instead
        self.predict_dataset = LoveDA(split="test", **self.kwargs)
