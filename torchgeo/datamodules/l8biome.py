# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L8 Biome datamodule."""

from typing import Any

import torch

from ..datasets import L8Biome, random_bbox_assignment
from .geo import GeoDataModule


class L8BiomeDataModule(GeoDataModule):
    """LightningDataModule implementation for the L8 Biome dataset.

    .. versionadded:: 0.5
    """

    mean = torch.tensor(0)
    std = torch.tensor(10000)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new L8BiomeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.L8Biome`.
        """
        super().__init__(L8Biome, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = L8Biome(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_assignment(dataset, [0.6, 0.2, 0.2], generator)
