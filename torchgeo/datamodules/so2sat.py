# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat datamodule."""

from typing import Any

import torch
from torch import Generator, Tensor
from torch.utils.data import random_split

from ..datasets import So2Sat
from .geo import NonGeoDataModule


class So2SatDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    If using the version 2 dataset, we use the train/val/test splits from the dataset.
    If using the version 3 datasets, we use a random 80/20 train/val split from the
    "train" set and use the "test" set as the test set.
    """

    means_per_version: dict[str, Tensor] = {
        '2': torch.tensor(
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
        '3_random': torch.tensor(
            [
                -0.00005541164581,
                -0.00001363245448,
                0.00004558943283,
                0.00002990907940,
                0.04451951629749,
                0.25862310103671,
                0.00032720731137,
                0.00123416595462,
                0.12428656593186,
                0.11001677362564,
                0.10230652367417,
                0.11532195526186,
                0.15989486018315,
                0.18204406482475,
                0.17513562590622,
                0.19565546643221,
                0.15648722649020,
                0.11122536338577,
            ]
        ),
        '3_block': torch.tensor(
            [
                -0.00004632368791,
                0.00001260869365,
                0.00005305557337,
                0.00003471369557,
                0.04449937686171,
                0.26046026815721,
                0.00087815394475,
                0.00086889627435,
                0.12381869777901,
                0.10944155483024,
                0.10176911573221,
                0.11465267892206,
                0.15870528223797,
                0.18053964470203,
                0.17366821871719,
                0.19390983961551,
                0.15536490486611,
                0.11057334452833,
            ]
        ),
    }
    means_per_version['3_culture_10'] = means_per_version['2']

    stds_per_version: dict[str, Tensor] = {
        '2': torch.tensor(
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
        '3_random': torch.tensor(
            [
                0.1756914,
                0.1761190,
                0.4600589,
                0.4563601,
                2.2492179,
                7.9056503,
                2.1917633,
                1.3148480,
                0.0392269,
                0.0470917,
                0.0653264,
                0.0624057,
                0.0758367,
                0.0891717,
                0.0905092,
                0.0996856,
                0.0990188,
                0.0873386,
            ]
        ),
        '3_block': torch.tensor(
            [
                0.1751797,
                0.1754073,
                0.4610124,
                0.4572122,
                0.8294254,
                7.1771026,
                0.9642598,
                0.8770835,
                0.0388311,
                0.0464986,
                0.0643833,
                0.0616141,
                0.0753004,
                0.0886178,
                0.0899500,
                0.0991759,
                0.0983276,
                0.0865943,
            ]
        ),
    }
    stds_per_version['3_culture_10'] = stds_per_version['2']

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        band_set: str = 'all',
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new So2SatDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            band_set: One of 'all', 's1', 's2', or 'rgb'.
            val_split_pct: Percentage of training data to use for validation in with
                the version 3 datasets.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.So2Sat`.

        .. versionadded:: 0.5
           The *val_split_pct* parameter, and the 'rgb' argument to *band_set*.
        """
        # https://github.com/Lightning-AI/lightning/issues/18616
        kwargs['version'] = str(kwargs.get('version', '2'))
        version = kwargs['version']
        kwargs['bands'] = So2Sat.BAND_SETS[band_set]
        self.val_split_pct = val_split_pct

        if band_set == 's1':
            self.mean = self.means_per_version[version][:8]
            self.std = self.stds_per_version[version][:8]
        elif band_set == 's2':
            self.mean = self.means_per_version[version][8:]
            self.std = self.stds_per_version[version][8:]
        elif band_set == 'rgb':
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
        if self.kwargs.get('version', '2') == '2':
            if stage in ['fit']:
                self.train_dataset = So2Sat(split='train', **self.kwargs)
            if stage in ['fit', 'validate']:
                self.val_dataset = So2Sat(split='validation', **self.kwargs)
            if stage in ['test']:
                self.test_dataset = So2Sat(split='test', **self.kwargs)
        else:
            if stage in ['fit', 'validate']:
                dataset = So2Sat(split='train', **self.kwargs)
                val_length = round(len(dataset) * self.val_split_pct)
                train_length = len(dataset) - val_length
                self.train_dataset, self.val_dataset = random_split(
                    dataset,
                    [train_length, val_length],
                    generator=Generator().manual_seed(0),
                )
            if stage in ['test']:
                self.test_dataset = So2Sat(split='test', **self.kwargs)
