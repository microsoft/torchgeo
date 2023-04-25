# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""C2Seg datamodule."""

from typing import Any

import torch

from ..datasets import C2Seg
from .geo import NonGeoDataModule


class C2SegDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the C2Seg dataset.

    .. versionadded:: 0.5
    """

    min = torch.tensor([0])
    max = torch.tensor([1])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new C2SegDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.C2Seg`.
        """
        super().__init__(C2Seg, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.train_dataset = C2Seg(split="train", **self.kwargs)
            self.val_dataset = C2Seg(split="val", **self.kwargs)
