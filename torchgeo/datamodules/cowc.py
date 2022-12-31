# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""COWC datamodule."""

from typing import Any, Optional

from kornia.augmentation import Normalize
from torch.utils.data import random_split

from ..datasets import COWCCounting
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class COWCCountingDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the COWC Counting dataset."""

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.COWCCounting`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            Normalize(mean=0, std=255), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Initialize the main Dataset objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        if self.kwargs.get("download", False):
            COWCCounting(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_val_dataset = COWCCounting(split="train", **self.kwargs)
        self.test_dataset = COWCCounting(split="test", **self.kwargs)
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset,
            [len(train_val_dataset) - len(self.test_dataset), len(self.test_dataset)],
        )
