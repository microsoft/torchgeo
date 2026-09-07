# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
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

    def test_getitem(self, dataset: BioMassters) -> None:
        sample = dataset[0]

        if dataset.as_time_series:
            assert sample['image'].ndim == 4
        else:
            assert sample['image'].ndim == 3
        if dataset.split == 'train':
            assert sample['mask'].ndim == 2
        else:
            assert 'mask' not in sample

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BioMassters(tmp_path)

    def test_plot(self, dataset: BioMassters) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        if dataset.split == 'train':
            sample['prediction'] = sample['mask'].unsqueeze(dim=0)
        dataset.plot(sample)
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()

    def test_plot_invalid_image_shape(self, dataset: BioMassters) -> None:
        with pytest.raises(ValueError, match='Expected image tensor'):
            dataset.plot({'image': torch.zeros(1)})

    def test_duplicate_monthly_acquisition(self) -> None:
        root = os.path.join('tests', 'data', 'biomassters')
        dataset = BioMassters(root, split='train', as_time_series=True)
        dataset.df = pd.concat([dataset.df, dataset.df.iloc[[0]]])

        with pytest.raises(ValueError, match='Expected one S1 acquisition per month'):
            dataset[0]
