# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SSL4EO datamodule."""

from typing import Any

import torch

from ..datasets import SSL4EOL, SSL4EOS12
from .geo import NonGeoDataModule


class SSL4EOLDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-L dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOLDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOL`.
        """
        super().__init__(SSL4EOL, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SSL4EOL(**self.kwargs)


class SSL4EOS12DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-S12 dataset.

    .. versionadded:: 0.5
    """

    # https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/datasets/EuroSat/eurosat_dataset.py#L97  # noqa: E501
    mean = torch.tensor(0)
    std = torch.tensor(10000)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOS12DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOS12`.
        """
        super().__init__(SSL4EOS12, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SSL4EOS12(**self.kwargs)
