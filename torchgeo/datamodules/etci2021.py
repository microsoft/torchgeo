# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ETCI 2021 datamodule."""

from typing import Any

import torch
from torch import Tensor

from ..datasets import ETCI2021
from .geo import NonGeoDataModule


class ETCI2021DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the ETCI2021 dataset.

    Splits the existing train split from the dataset into train/val with 80/20
    proportions, then uses the existing val dataset as the test data.

    .. versionadded:: 0.2
    """

    mean = torch.tensor(
        [
            128.02253931,
            128.02253931,
            128.02253931,
            128.11221701,
            128.11221701,
            128.11221701,
        ]
    )
    std = torch.tensor(
        [89.8145088, 89.8145088, 89.8145088, 95.2797861, 95.2797861, 95.2797861]
    )

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new ETCI2021DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ETCI2021`.
        """
        super().__init__(ETCI2021, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = ETCI2021(split="train", **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = ETCI2021(split="val", **self.kwargs)
        if stage in ["predict"]:
            # Test set masks are not public, use for prediction instead
            self.predict_dataset = ETCI2021(split="test", **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if not self.trainer.predicting:
                # Evaluate against flood mask, not water mask
                batch["mask"] = (batch["mask"][:, 1] > 0).long()

        return super().on_after_batch_transfer(batch, dataloader_idx)
