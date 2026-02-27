# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import (
    DatasetNotFoundError,
    HabitAlp2,
    HabitAlp2CD,
    IntersectionDataset,
    UnionDataset,
)


class TestHabitAlp2:
    @pytest.fixture(
        params=product(
            ['2003', '2013', '2020'],
            [None, ('R', 'G', 'B'), ('R', 'G', 'B', 'NIR'), ('R', 'G', 'B', 'ndsm')],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> HabitAlp2:
        year, bands = request.param

        if year == '2003':
            if bands is not None and any(b not in ['R', 'G', 'B'] for b in bands):
                pytest.skip('2003 only has RGB bands')

        src = os.path.join('tests', 'data', 'habitalp')
        for folder in ['data_2003', 'data_2013', 'data_2020', 'labels']:
            src_folder = os.path.join(src, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)

        root = tmp_path
        return HabitAlp2(root, year=year, bands=bands, transforms=nn.Identity())

    def test_getitem(self, dataset: HabitAlp2) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].dtype == torch.float32
        assert x['mask'].dtype == torch.int64

    def test_len(self, dataset: HabitAlp2) -> None:
        assert len(dataset) >= 1

    def test_and(self, dataset: HabitAlp2) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: HabitAlp2) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: HabitAlp2) -> None:
        HabitAlp2(root=dataset.root, year=dataset.year)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            HabitAlp2(tmp_path)

    def test_plot(self, dataset: HabitAlp2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: HabitAlp2) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: HabitAlp2) -> None:
        with pytest.raises(IndexError, match=r'not found in .* with bounds:'):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='year must be one of'):
            HabitAlp2(tmp_path, year='1999')

    def test_invalid_bands(self, tmp_path: Path) -> None:
        src = os.path.join('tests', 'data', 'habitalp')
        for folder in ['data_2003', 'labels']:
            src_folder = os.path.join(src, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)

        with pytest.raises(AssertionError, match='not available for year'):
            HabitAlp2(tmp_path, year='2003', bands=('NIR',))


class TestHabitAlp2CD:
    @pytest.fixture(
        params=product(
            ['2003_2013', '2013_2020'], [None, ('R', 'G', 'B'), ('R', 'G', 'B', 'NIR')]
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> HabitAlp2CD:
        pair, bands = request.param

        if pair == '2003_2013':
            if bands is not None and 'NIR' in bands:
                pytest.skip('2003_2013 pair only supports RGB bands')

        src = os.path.join('tests', 'data', 'habitalp')
        for folder in ['data_2003', 'data_2013', 'data_2020', 'labels']:
            src_folder = os.path.join(src, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)

        root = tmp_path
        return HabitAlp2CD(root, pair=pair, bands=bands, transforms=nn.Identity())

    def test_getitem(self, dataset: HabitAlp2CD) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].dtype == torch.float32
        assert x['mask'].dtype == torch.int64
        assert x['image'].ndim == 4
        assert x['image'].shape[0] == 2
        assert x['mask'].ndim == 3
        assert x['mask'].shape[0] == 1

    def test_len(self, dataset: HabitAlp2CD) -> None:
        assert len(dataset) >= 1

    def test_and(self, dataset: HabitAlp2CD) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: HabitAlp2CD) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: HabitAlp2CD) -> None:
        HabitAlp2CD(root=dataset.root, pair=dataset.pair)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            HabitAlp2CD(tmp_path)

    def test_plot(self, dataset: HabitAlp2CD) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_invalid_query(self, dataset: HabitAlp2CD) -> None:
        with pytest.raises(IndexError, match=r'not found in .* with bounds:'):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_invalid_pair(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='pair must be one of'):
            HabitAlp2CD(tmp_path, pair='2003_2020')

    def test_invalid_bands(self, tmp_path: Path) -> None:
        src = os.path.join('tests', 'data', 'habitalp')
        for folder in ['data_2003', 'data_2013', 'labels']:
            src_folder = os.path.join(src, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)

        with pytest.raises(AssertionError, match='not available for pair'):
            HabitAlp2CD(tmp_path, pair='2003_2013', bands=('NIR',))
