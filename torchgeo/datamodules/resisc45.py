# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 datamodule."""

from typing import Any, Optional

import kornia.augmentation as K

from ..datasets import RESISC45
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class RESISC45DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = [127.86820969, 127.88083247, 127.84341029]
    band_stds = [51.8668062, 47.2380768, 47.0613924]

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.RESISC45`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.band_means, std=self.band_stds),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(
                p=0.5,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                silence_instantiation_warning=True,
            ),
            data_keys=["image"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.band_means, std=self.band_stds), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            RESISC45(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = RESISC45(split="train", **self.kwargs)
        self.val_dataset = RESISC45(split="val", **self.kwargs)
        self.test_dataset = RESISC45(split="test", **self.kwargs)
