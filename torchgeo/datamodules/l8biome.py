# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L8 Biome datamodule."""

from typing import Any, Tuple, Union

import torch

from ..datasets import L8Biome, random_bbox_assignment
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from .geo import GeoDataModule


class L8BiomeDataModule(GeoDataModule):
    """LightningDataModule implementation for the L8 Biome dataset.

    .. versionadded:: 0.5
    """

    mean = torch.tensor(0)
    std = torch.tensor(10000)

    def __init__(
        self,
        batch_size: int = 1,
        patch_size: Union[int, Tuple[int, int]] = 32,
        length: int = 5,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new L8BiomeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.L8Biome`.
        """
        super().__init__(
            L8Biome,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

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

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )
