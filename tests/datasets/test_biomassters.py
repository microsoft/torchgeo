# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datasets import BioMassters, DatasetNotFoundError


class TestBioMassters:
    @pytest.fixture(
        params=product(['train', 'test'], [['S1'], ['S2'], ['S1', 'S2']], [True, False])
    )
    def dataset(self, request: SubRequest) -> BioMassters:
        root = os.path.join('tests', 'data', 'biomassters')
        split, sensors, as_time_series = request.param
        return BioMassters(
            root, split=split, sensors=sensors, as_time_series=as_time_series
        )

    def test_len_of_ds(self, dataset: BioMassters) -> None:
        assert len(dataset) > 0

    def test_invalid_split(self, dataset: BioMassters) -> None:
        with pytest.raises(AssertionError):
            BioMassters(dataset.root, split='foo')

    def test_invalid_bands(self, dataset: BioMassters) -> None:
        with pytest.raises(AssertionError):
            BioMassters(dataset.root, sensors=['S3'])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BioMassters(tmp_path)

    def test_plot(self, dataset: BioMassters) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        if dataset.split == 'train':
            sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
