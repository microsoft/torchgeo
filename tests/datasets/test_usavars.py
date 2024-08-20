# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import DatasetNotFoundError, USAVars


class TestUSAVars:
    @pytest.fixture(
        params=zip(
            ['train', 'val', 'test'],
            [
                ['elevation', 'population', 'treecover'],
                ['elevation', 'population'],
                ['treecover'],
            ],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> USAVars:
        md5 = 'b504580a00bdc27097d5421dec50481b'
        monkeypatch.setattr(USAVars, 'md5', md5)

        data_url = os.path.join('tests', 'data', 'usavars', 'uar.zip')
        monkeypatch.setattr(USAVars, 'data_url', data_url)

        label_urls = {
            'elevation': os.path.join('tests', 'data', 'usavars', 'elevation.csv'),
            'population': os.path.join('tests', 'data', 'usavars', 'population.csv'),
            'treecover': os.path.join('tests', 'data', 'usavars', 'treecover.csv'),
            'income': os.path.join('tests', 'data', 'usavars', 'income.csv'),
            'nightlights': os.path.join('tests', 'data', 'usavars', 'nightlights.csv'),
            'roads': os.path.join('tests', 'data', 'usavars', 'roads.csv'),
            'housing': os.path.join('tests', 'data', 'usavars', 'housing.csv'),
        }
        monkeypatch.setattr(USAVars, 'label_urls', label_urls)

        split_metadata = {
            'train': {
                'url': os.path.join('tests', 'data', 'usavars', 'train_split.txt'),
                'filename': 'train_split.txt',
                'md5': 'b94f3f6f63110b253779b65bc31d91b5',
            },
            'val': {
                'url': os.path.join('tests', 'data', 'usavars', 'val_split.txt'),
                'filename': 'val_split.txt',
                'md5': 'e39aa54b646c4c45921fcc9765d5a708',
            },
            'test': {
                'url': os.path.join('tests', 'data', 'usavars', 'test_split.txt'),
                'filename': 'test_split.txt',
                'md5': '4ab0f5549fee944a5690de1bc95ed245',
            },
        }
        monkeypatch.setattr(USAVars, 'split_metadata', split_metadata)

        root = tmp_path
        split, labels = request.param
        transforms = nn.Identity()

        return USAVars(
            root, split, labels, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: USAVars) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].ndim == 3
        assert len(x.keys()) == 4  # image, labels, centroid_lat, centroid_lon
        assert x['image'].shape[0] == 4  # R, G, B, Inf
        assert len(dataset.labels) == len(x['labels'])
        assert len(x['centroid_lat']) == 1
        assert len(x['centroid_lon']) == 1

    def test_len(self, dataset: USAVars) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        elif dataset.split == 'val':
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_add(self, dataset: USAVars) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: USAVars) -> None:
        USAVars(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'usavars', 'uar.zip')
        root = tmp_path
        shutil.copy(pathname, root)
        csvs = [
            'elevation.csv',
            'population.csv',
            'treecover.csv',
            'income.csv',
            'nightlights.csv',
            'roads.csv',
            'housing.csv',
        ]
        for csv in csvs:
            shutil.copy(os.path.join('tests', 'data', 'usavars', csv), root)
        splits = ['train_split.txt', 'val_split.txt', 'test_split.txt']
        for split in splits:
            shutil.copy(os.path.join('tests', 'data', 'usavars', split), root)

        USAVars(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            USAVars(tmp_path)

    def test_plot(self, dataset: USAVars) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()
