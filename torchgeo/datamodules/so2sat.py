# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat datamodule."""

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import So2Sat
from .geo import NonGeoDataModule


class So2SatDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    If using the version "2" dataset, we use the train/val/test splits from the dataset.
    If using the version "3" datasets, we use a random 80/20 train/val split from the
    "train" set and use the "test" set as the test set.
    """

    means_per_version: dict[str, Tensor] = {
        "2": torch.tensor(
            [
                -0.00003591224260,
                -0.00000765856128,
                0.00005937385750,
                0.00002516623150,
                0.04420110660000,
                0.25761027100000,
                0.00075567433700,
                0.00135034668000,
                0.12375696117681,
                0.10927746363683,
                0.10108552032678,
                0.11423986161140,
                0.15926566920230,
                0.18147236008771,
                0.17457403122913,
                0.19501607349635,
                0.15428468872076,
                0.10905050699570,
            ]
        ),
        "3_random": torch.tensor([]),
        "3_block": torch.tensor([]),
        "3_culture_10": torch.tensor(  # note: this is the same as "2"
            [
                -0.00003591224260,
                -0.00000765856128,
                0.00005937385750,
                0.00002516623150,
                0.04420110660000,
                0.25761027100000,
                0.00075567433700,
                0.00135034668000,
                0.12375696117681,
                0.10927746363683,
                0.10108552032678,
                0.11423986161140,
                0.15926566920230,
                0.18147236008771,
                0.17457403122913,
                0.19501607349635,
                0.15428468872076,
                0.10905050699570,
            ]
        ),
    }

    stds_per_version: dict[str, Tensor] = {
        "2": torch.tensor(
            [
                0.17555201,
                0.17556463,
                0.45998793,
                0.45598876,
                2.85599092,
                8.32480061,
                2.44987574,
                1.46473530,
                0.03958795,
                0.04777826,
                0.06636616,
                0.06358874,
                0.07744387,
                0.09101635,
                0.09218466,
                0.10164581,
                0.09991773,
                0.08780632,
            ]
        ),
        "3_random": torch.tensor([]),
        "3_block": torch.tensor([]),
        "3_culture_10": torch.tensor(  # note: this is the same as "2"
            [
                0.17555201,
                0.17556463,
                0.45998793,
                0.45598876,
                2.85599092,
                8.32480061,
                2.44987574,
                1.46473530,
                0.03958795,
                0.04777826,
                0.06636616,
                0.06358874,
                0.07744387,
                0.09101635,
                0.09218466,
                0.10164581,
                0.09991773,
                0.08780632,
            ]
        ),
    }

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        version: str = "2",
        band_set: str = "all",
        **kwargs: Any,
    ) -> None:
        """Initialize a new So2SatDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            version: One of "2" or "3_random", "3_block", or "3_culture_10".
            num_workers: Number of workers for parallel data loading.
            band_set: One of 'all', 's1', 's2', or 'rgb'.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.So2Sat`.

        .. versionadded:: 0.5
           The *version* parameter and the 'rgb' argument to *band_set*.
        """
        kwargs["bands"] = So2Sat.BAND_SETS[band_set]
        kwargs["version"] = version
        self.version = version

        if band_set == "s1":
            self.mean = self.means_per_version[version][:8]
            self.std = self.stds_per_version[version][:8]
        elif band_set == "s2":
            self.mean = self.means_per_version[version][8:]
            self.std = self.stds_per_version[version][8:]
        elif band_set == "rgb":
            self.mean = self.means_per_version[version][[10, 9, 8]]
            self.std = self.stds_per_version[version][[10, 9, 8]]

        super().__init__(So2Sat, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self.version == "2":
            if stage in ["fit"]:
                self.train_dataset = So2Sat(split="train", **self.kwargs)
            if stage in ["fit", "validate"]:
                self.val_dataset = So2Sat(split="validation", **self.kwargs)
            if stage in ["test"]:
                self.test_dataset = So2Sat(split="test", **self.kwargs)
        else:
            if stage in ["fit", "validate"]:
                dataset = So2Sat(split="train", **self.kwargs)
                self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])
            if stage in ["test"]:
                self.test_dataset = So2Sat(split="test", **self.kwargs)
