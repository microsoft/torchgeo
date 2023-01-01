# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai datamodule."""

from typing import Any, Optional

import kornia.augmentation as K

from ..datasets import LandCoverAI
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class LandCoverAIDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LandCover.ai dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LandCoverAI`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0),
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
        self.aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0), data_keys=["image", "mask"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            LandCoverAI(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = LandCoverAI(split="train", **self.kwargs)
        self.val_dataset = LandCoverAI(split="val", **self.kwargs)
        self.test_dataset = LandCoverAI(split="test", **self.kwargs)
