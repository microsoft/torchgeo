# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, WorldStrat


class TestWorldStrat:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> WorldStrat:
        url = os.path.join('tests', 'data', 'worldstrat')

        file_info_dict = {
            'hr_dataset': {
                'url': os.path.join(url, 'hr_dataset.tar.gz'),
                'filename': 'hr_dataset.tar.gz',
                'md5': '09a5666bcbb9d2ad3e5efd4cf5db36f7',
            },
            'lr_dataset_l1c': {
                'url': os.path.join(url, 'lr_dataset_l1c.tar.gz'),
                'filename': 'lr_dataset_l1c.tar.gz',
                'md5': 'd7073eca67e06e29ce1cf30198c6dbfe',
            },
            'lr_dataset_l2a': {
                'url': os.path.join(url, 'lr_dataset_l2a.tar.gz'),
                'filename': 'lr_dataset_l2a.tar.gz',
                'md5': 'fbce39a8d389b63bef5968577d93d3eb',
            },
            'metadata': {
                'url': os.path.join(url, 'metadata.csv'),
                'filename': 'metadata.csv',
                'md5': 'd18a9de2a1fcc6eda95d0975d8ce12f1',
            },
            'train_val_test_split': {
                'url': os.path.join(url, 'stratified_train_val_test_split.csv'),
                'filename': 'stratified_train_val_test_split.csv',
                'md5': 'c6941d2c0f044d716ea5f0ab4277cba6',
            },
        }
        monkeypatch.setattr(WorldStrat, 'file_info_dict', file_info_dict)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return WorldStrat(
            root, split=split, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: WorldStrat) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        for modality in dataset.modalities:
            assert isinstance(x[f'image_{modality}'], torch.Tensor)

    def test_len(self, dataset: WorldStrat) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: WorldStrat) -> None:
        WorldStrat(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        file_list = [
            'hr_dataset.tar.gz',
            'lr_dataset_l1c.tar.gz',
            'lr_dataset_l2a.tar.gz',
            'metadata.csv',
            'stratified_train_val_test_split.csv',
        ]
        dir = os.path.join('tests', 'data', 'worldstrat')
        for filename in file_list:
            shutil.copyfile(
                os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
            )
        WorldStrat(root=str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            WorldStrat(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            WorldStrat(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'hr_dataset.tar.gz'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Archive'):
            WorldStrat(root=tmp_path, checksum=True)

    def test_plot(self, dataset: WorldStrat) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()
