# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Marine Debris datamodule."""

from typing import Any, Optional

from kornia.augmentation import Normalize

from ..datasets import NASAMarineDebris
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule
from .utils import dataset_split


class NASAMarineDebrisDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the NASA Marine Debris dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NASAMarineDebris`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            Normalize(mean=0.0 std=255.0), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            NASAMarineDebris(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.dataset = NASAMarineDebris(**self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )
