# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Functional Map of the World datamodule."""

from typing import Any

from ..datasets import FMoW
from .geo import NonGeoDataModule
from .utils import collate_fn_detection


class FMoWDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the fMoW dataset.

    .. versionadded:: 0.11
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new FMoWDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FMoW`.
        """
        super().__init__(FMoW, batch_size, num_workers, **kwargs)
        self.collate_fn = collate_fn_detection
