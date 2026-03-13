# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Any

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

DATA_DIR = os.path.join('tests', 'data', 'habitalp')


def _copy_file(url: str, root: Path, filename: str, **kwargs: Any) -> None:
    dst = os.path.join(str(root), filename)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(url, dst)


class TestHabitAlp2:
    @pytest.fixture(params=['2003', '2013', '2020'])
    def dataset(self, tmp_path: Path, request: SubRequest) -> HabitAlp2:
        for folder in ['data_2003', 'data_2013', 'data_2020', 'labels']:
            src_folder = os.path.join(DATA_DIR, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)
        return HabitAlp2(tmp_path, year=request.param, transforms=nn.Identity())

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
        for folder in ['data_2003', 'labels']:
            src_folder = os.path.join(DATA_DIR, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)
        with pytest.raises(AssertionError, match='not available for year'):
            HabitAlp2(tmp_path, year='2003', bands=('NIR',))

    @pytest.mark.parametrize('bands', [('R', 'dtm'), ('NIR', 'dtm')])
    def test_download(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, bands: tuple[str, ...]
    ) -> None:
        monkeypatch.setattr(HabitAlp2, 'url', DATA_DIR + '/')
        monkeypatch.setattr('torchgeo.datasets.habitalp2.download_url', _copy_file)
        HabitAlp2(tmp_path, download=True, year='2013', bands=bands)


class TestHabitAlp2CD:
    @pytest.fixture(params=['2003_2013', '2013_2020'])
    def dataset(self, tmp_path: Path, request: SubRequest) -> HabitAlp2CD:
        for folder in ['data_2003', 'data_2013', 'data_2020', 'labels']:
            src_folder = os.path.join(DATA_DIR, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)
        return HabitAlp2CD(tmp_path, pair=request.param, transforms=nn.Identity())

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

    @pytest.fixture
    def multiclass_dataset(self, tmp_path: Path) -> HabitAlp2CD:
        for folder in ['data_2013', 'data_2020', 'labels']:
            shutil.copytree(
                os.path.join(DATA_DIR, folder), os.path.join(tmp_path, folder)
            )
        return HabitAlp2CD(tmp_path, task='multiclass')

    def test_getitem_multiclass(self, multiclass_dataset: HabitAlp2CD) -> None:
        x = multiclass_dataset[multiclass_dataset.bounds]
        assert x['mask'].ndim == 3
        assert x['mask'].shape[0] == 1

    def test_plot_multiclass(self, multiclass_dataset: HabitAlp2CD) -> None:
        multiclass_dataset.plot(
            multiclass_dataset[multiclass_dataset.bounds], suptitle='Test'
        )
        plt.close()

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
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: HabitAlp2CD) -> None:
        with pytest.raises(IndexError, match=r'not found in .* with bounds:'):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_invalid_pair(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='pair must be one of'):
            HabitAlp2CD(tmp_path, pair='2003_2020')

    def test_invalid_task(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='task must be one of'):
            HabitAlp2CD(tmp_path, task='segmentation')

    def test_invalid_bands(self, tmp_path: Path) -> None:
        for folder in ['data_2003', 'data_2013', 'labels']:
            src_folder = os.path.join(DATA_DIR, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)
        with pytest.raises(AssertionError, match='not available for pair'):
            HabitAlp2CD(tmp_path, pair='2003_2013', bands=('NIR',))

    def _setup_cd_without_change_mask(self, tmp_path: Path) -> None:
        for folder in ['data_2013', 'data_2020']:
            shutil.copytree(
                os.path.join(DATA_DIR, folder), os.path.join(tmp_path, folder)
            )
        os.makedirs(os.path.join(tmp_path, 'labels'))
        for f in ['classes_2013.tif', 'classes_2020.tif']:
            shutil.copy(
                os.path.join(DATA_DIR, 'labels', f), os.path.join(tmp_path, 'labels', f)
            )

    def test_not_downloaded_change_mask(self, tmp_path: Path) -> None:
        self._setup_cd_without_change_mask(tmp_path)
        with pytest.raises(DatasetNotFoundError):
            HabitAlp2CD(tmp_path, pair='2013_2020')

    def test_pair_as_integer(self, tmp_path: Path) -> None:
        for folder in ['data_2013', 'data_2020', 'labels']:
            shutil.copytree(
                os.path.join(DATA_DIR, folder), os.path.join(tmp_path, folder)
            )
        HabitAlp2CD(tmp_path, pair='20132020')

    def test_download_change_mask(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup_cd_without_change_mask(tmp_path)
        monkeypatch.setattr(HabitAlp2CD, 'url', DATA_DIR + '/')
        monkeypatch.setattr('torchgeo.datasets.habitalp2.download_url', _copy_file)
        HabitAlp2CD(tmp_path, pair='2013_2020', download=True)
