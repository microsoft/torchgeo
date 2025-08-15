# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import shutil
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, AI4ArcticSeaIce

pytest.importorskip('xarray', minversion='2023.9')
pytest.importorskip('netCDF4', minversion='1.5.4')

valid_amsr2_vars = ('btemp_6_9h', 'btemp_6_9v', 'btemp_7_3h', 'btemp_7_3v')
valid_weather_vars = ('u10m_rotated', 'v10m_rotated')


class TestAI4ArcticSeaIce:
    @pytest.fixture(
        params=zip(
            ['train', 'train', 'test', 'test'],
            ['SOD', 'SIC', 'FLOE', 'SIC'],
            [None, 'distance_map', None, 'distance_map'],
            [valid_amsr2_vars, None, valid_amsr2_vars, None],
            [valid_weather_vars, None, valid_weather_vars, None],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> AI4ArcticSeaIce:
        url = os.path.join('tests', 'data', 'ai4arctic_sea_ice', '{}')
        monkeypatch.setattr(AI4ArcticSeaIce, 'url', url)
        files = [
            {'name': 'train.tar.gzaa', 'md5': '399952b2603d0d508a30909357e6956a'},
            {'name': 'train.tar.gzab', 'md5': 'a998c852a2f418394f97cb1f99716489'},
            {'name': 'test.tar.gz', 'md5': 'b81e53b4c402a64d53854f02f66ce938'},
            {'name': 'metadata.csv', 'md5': 'd1222877af76d3fe9620678c930d70f0'},
        ]
        monkeypatch.setattr(AI4ArcticSeaIce, 'files', files)

        monkeypatch.setattr(AI4ArcticSeaIce, 'valid_amsr2_vars', valid_amsr2_vars)

        monkeypatch.setattr(AI4ArcticSeaIce, 'valid_weather_vars', valid_weather_vars)
        root = tmp_path
        split, target_var, geo_var, amsr2_var, weather_var = request.param
        transforms = nn.Identity()
        return AI4ArcticSeaIce(
            root,
            split=split,
            target_var=target_var,
            geo_var=geo_var,
            amsr2_vars=amsr2_var,
            weather_vars=weather_var,
            transforms=transforms,
            download=True,
            checksum=False,
        )

    def test_getitem(self, dataset: AI4ArcticSeaIce) -> None:
        x = dataset[0]
        assert isinstance(x, dict)

    def test_len(self, dataset: AI4ArcticSeaIce) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AI4ArcticSeaIce(tmp_path)

    def test_already_downloaded_and_extracted(self, dataset: AI4ArcticSeaIce) -> None:
        AI4ArcticSeaIce(root=dataset.root, download=False)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            AI4ArcticSeaIce(split='foo')

    def test_plot(self, dataset: AI4ArcticSeaIce) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask'])
        dataset.plot(sample, suptitle='Test with prediction')
        plt.close()
