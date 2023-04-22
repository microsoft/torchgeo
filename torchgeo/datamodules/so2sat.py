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

    # TODO: calculate mean/std dev of s1 bands
    mean = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
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

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        band_set: str = "all",
        **kwargs: Any,
    ) -> None:
        """Initialize a new So2SatDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            band_set: One of 'all', 's1', or 's2'.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.So2Sat`.
        """
        kwargs["bands"] = So2Sat.BAND_SETS[band_set]

        if band_set == "s1":
            self.mean = self.mean[:8]
            self.std = self.std[:8]
        elif band_set == "s2":
            self.mean = self.mean[8:]
            self.std = self.std[8:]

        super().__init__(So2Sat, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = So2Sat(split="train", **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = So2Sat(split="validation", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = So2Sat(split="test", **self.kwargs)
