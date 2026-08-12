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
    @pytest.fixture(autouse=True)
    def fake_file_info(self, monkeypatch: MonkeyPatch) -> None:
        url = os.path.join('tests', 'data', 'worldstrat')

        file_info_dict = {
            'hr_dataset': {
                'url': os.path.join(url, 'hr_dataset.zip'),
                'filename': 'hr_dataset.zip',
                'sha256': '44b60bc03ad45281886c65c85c6c09f3aa2931d1f154ac985470eb0329e4085b',
            },
            'lr_dataset_l1c': {
                'url': os.path.join(url, 'lr_dataset_l1c.zip'),
                'filename': 'lr_dataset_l1c.zip',
                'sha256': '6e29fe2d2ea65c3a1269f0344f61f2f1eb0e01a2eff64f6a9dd26745d10b6e90',
            },
            'lr_dataset_l2a': {
                'url': os.path.join(url, 'lr_dataset_l2a.zip'),
                'filename': 'lr_dataset_l2a.zip',
                'sha256': '6f39eedc1c0aff78dd4bb8f69c26f87993c86e19411ebd24f486babfa875a545',
            },
            'metadata': {
                'url': os.path.join(url, 'metadata.csv'),
                'filename': 'metadata.csv',
                'sha256': 'bfc8f0ab3dc48617d83146eefd609d03ae41e533bc5009476289708d6892d692',
            },
            'train_val_test_split': {
                'url': os.path.join(url, 'stratified_train_val_test_split.csv'),
                'filename': 'stratified_train_val_test_split.csv',
                'sha256': '776653b439385ea7061bfaf27d57eabebceb81c66479e76de67fa3f02ef82539',
            },
        }
        monkeypatch.setattr(WorldStrat, 'file_info_dict', file_info_dict)

    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(self, tmp_path: Path, request: SubRequest) -> WorldStrat:
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return WorldStrat(
            root, split=split, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: WorldStrat) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert all(isinstance(value, torch.Tensor) for value in x.values())

        for modality in dataset.modalities:
            assert x[f'image_{modality}'].dtype == torch.float32

        # one low-res date per timestep, aligned to the stacked time dimension
        low_res_date = x['low_res_date']
        assert low_res_date.dtype == torch.float64
        assert low_res_date.ndim == 1
        assert len(low_res_date) == x['image_l1c'].shape[0]
        assert torch.equal(low_res_date, low_res_date.sort().values)

        # remaining metadata is constant across a tile's rows
        for key in ('lon', 'lat', 'high_res_date'):
            assert x[key].dtype == torch.float64
            assert x[key].ndim == 0

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
