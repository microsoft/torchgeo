# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""HySpecNet datamodule."""

from typing import Any

import torch

from ..datasets import HySpecNet11k
from .geo import NonGeoDataModule


class HySpecNet11kDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the HySpecNet11k dataset.

    .. versionadded:: 0.7
    """

    # https://git.tu-berlin.de/rsim/hyspecnet-tools/-/blob/main/tif_to_npy.ipynb
    mean = torch.tensor(0)
    std = torch.tensor(10000)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new HySpecNet11kDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.HySpecNet11k`.
        """
        super().__init__(HySpecNet11k, batch_size, num_workers, **kwargs)
