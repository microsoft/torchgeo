# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SSL4EO datamodule."""

from typing import Any

from ..datasets import SSL4EOLBenchmark
from .geo import NonGeoDataModule


class SSL4EOLBenchmarkDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-L dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOLDownstreamDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOL`.
        """
        super().__init__(SSL4EOLBenchmark, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SSL4EOLBenchmark(**self.kwargs)
