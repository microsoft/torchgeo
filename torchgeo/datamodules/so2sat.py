# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat datamodule."""

from typing import Any

import torch

from ..datasets import So2Sat
from .geo import NonGeoDataModule


class So2SatDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    Uses the train/val/test splits from the dataset.
    """

    mean = torch.tensor(
        [
            0.12375696117681859,
            0.1092774636368323,
            0.1010855203267882,
            0.1142398616114001,
            0.1592656692023089,
            0.18147236008771792,
            0.1745740312291377,
            0.19501607349635292,
            0.15428468872076637,
            0.10905050699570007,
        ]
    )

    std = torch.tensor(
        [
            0.03958795985905458,
            0.047778262752410296,
            0.06636616706371974,
            0.06358874912497474,
            0.07744387147984592,
            0.09101635085921553,
            0.09218466562387101,
            0.10164581233948201,
            0.09991773043519253,
            0.08780632509122865,
        ]
    )

    # this reorders the bands to put S2 RGB first, then remainder of S2
    reindex_to_rgb_first = [2, 1, 0, 3, 4, 5, 6, 7, 8, 9]

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new So2SatDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.So2Sat`
        """
        super().__init__(So2Sat, batch_size, num_workers, **kwargs)
