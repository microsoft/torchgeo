# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datamodules."""

from typing import Any, Optional

import kornia.augmentation as K

from ..datasets import SpaceNet1
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule
from .utils import dataset_split


class SpaceNet1DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SpaceNet1 dataset.

    Randomly splits into train/val/test.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet1.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SpaceNet1`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.kwargs = kwargs

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=0, std=255),
            K.PadTo((448, 448)),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(
                p=0.5,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                silence_instantiation_warning=True,
            ),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=0, std=255),
            K.PadTo((448, 448)),
            data_keys=["image", "mask"],
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            SpaceNet1(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.dataset = SpaceNet1(**self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )
