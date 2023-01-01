# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars datamodule."""

from typing import Any, Optional

import kornia.augmentation as K

from ..datasets import USAVars
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class USAVarsDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the USAVars dataset.

    Uses random train/val/test splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for USAVars based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.USAVars`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            USAVars(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.
        """
        self.train_dataset = USAVars(split="train", **self.kwargs)
        self.val_dataset = USAVars(split="val", **self.kwargs)
        self.test_dataset = USAVars(split="test", **self.kwargs)
