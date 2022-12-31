# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""xView2 datamodule."""

from typing import Any, Optional

from kornia.augmentation import Normalize

from ..datasets import XView2
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule
from .utils import dataset_split


class XView2DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the xView2 dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for xView2 based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.XView2`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            Normalize(mean=0.0 std=255.0), data_keys=["image"]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        dataset = XView2(split="train", **self.kwargs)
        self.train_dataset, self.val_dataset = dataset_split(
            dataset, val_pct=self.val_split_pct
        )
        self.test_dataset = XView2(split="test", **self.kwargs)
