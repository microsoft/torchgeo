# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

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
                'sha256': 'b1692b15ae6633334ebe456cdf7fc419cf285207ef66ca0d6e05b107f254422b',
            },
            'lr_dataset_l1c': {
                'url': os.path.join(url, 'lr_dataset_l1c.zip'),
                'filename': 'lr_dataset_l1c.zip',
                'sha256': '4c96167b74dfd2b86e848d861912077d3b50c6d1d53d25e7fa50738ab198d368',
            },
            'lr_dataset_l2a': {
                'url': os.path.join(url, 'lr_dataset_l2a.zip'),
                'filename': 'lr_dataset_l2a.zip',
                'sha256': 'cc991961f9c2f8293e0205685bd6751172bd3c19819b7ae2220b97bb21c4b694',
            },
            'metadata': {
                'url': os.path.join(url, 'metadata.csv'),
                'filename': 'metadata.csv',
                'sha256': '080b8e7ecbef10454047ca1e439eb45aefba2f41ba3bfd4dc238e3f9d7a08ea1',
            },
            'train_val_test_split': {
                'url': os.path.join(url, 'stratified_train_val_test_split.csv'),
                'filename': 'stratified_train_val_test_split.csv',
                'sha256': '776653b439385ea7061bfaf27d57eabebceb81c66479e76de67fa3f02ef82539',
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
