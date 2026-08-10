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
                'url': os.path.join(url, 'hr_dataset.zip'),
                'filename': 'hr_dataset.zip',
                'md5': '531a2262b55985a9af1d99a7ee890cc2',
            },
            'lr_dataset_l1c': {
                'url': os.path.join(url, 'lr_dataset_l1c.zip'),
                'filename': 'lr_dataset_l1c.zip',
                'md5': 'dbf5882a22c751c5fe0822c5a4db06d4',
            },
            'lr_dataset_l2a': {
                'url': os.path.join(url, 'lr_dataset_l2a.zip'),
                'filename': 'lr_dataset_l2a.zip',
                'md5': 'f48ad8b9dc79afa87f44d0d89d2c6544',
            },
            'metadata': {
                'url': os.path.join(url, 'metadata.csv'),
                'filename': 'metadata.csv',
                'md5': '84492378455f689a49078e187dfdf0b6',
            },
            'train_val_test_split': {
                'url': os.path.join(url, 'stratified_train_val_test_split.csv'),
                'filename': 'stratified_train_val_test_split.csv',
                'md5': 'c9eb98a9a45a57ef6028a6ef8102485d',
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
            assert x[f'image_{modality}'].dtype == torch.float32

        # one low-res date per timestep, aligned to the stacked time dimension
        assert isinstance(x['low_res_date'], list)
        assert len(x['low_res_date']) == x['image_l1c'].shape[0]
        assert x['low_res_date'] == sorted(x['low_res_date'])

        # remaining metadata is constant across a tile's rows
        for key in ('lon', 'lat', 'high_res_date'):
            assert not isinstance(x[key], list)

    def test_sentinel_paths_sorted_by_index(self, dataset: WorldStrat) -> None:
        aoi = dataset.file_path_df['tile'][0]
        data_dir = os.path.join(dataset.root, dataset.lr_dir, aoi, 'L1C')
        pairs = dataset._sentinel_paths(data_dir)

        assert [n for n, _ in pairs] == [1, 2, 3, 4]
        for n, path in pairs:
            assert os.path.basename(path) == f'{aoi}-{n}-L1C_data.tiff'

    def test_len(self, dataset: WorldStrat) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: WorldStrat) -> None:
        WorldStrat(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        file_list = [
            'hr_dataset.zip',
            'lr_dataset_l1c.zip',
            'lr_dataset_l2a.zip',
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
        with open(os.path.join(tmp_path, 'hr_dataset.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Archive'):
            WorldStrat(root=tmp_path, checksum=True)

    def test_plot(self, dataset: WorldStrat) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

    def test_pred_plot(self, dataset: WorldStrat) -> None:
        x = dataset[0]
        x['prediction'] = x['image_hr_rgbn']
        dataset.plot(x, suptitle='Test')
        plt.close()
