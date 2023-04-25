# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""WHISPERS23 datamodule."""

from typing import Any

import torch

from ..datasets import WHISPERS23
from .geo import NonGeoDataModule


class WHISPERS23DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the WHISPERS23 dataset.

    .. versionadded:: 0.5
    """

    min = torch.tensor([0])
    max = torch.tensor([1])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new WHISPERS23DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.WHISPERS23`.
        """
        super().__init__(WHISPERS23, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.train_dataset = WHISPERS23(split="train", **self.kwargs)
            self.val_dataset = WHISPERS23(split="val", **self.kwargs)
