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

from torchgeo.datasets import MDAS, DatasetNotFoundError


class TestMDAS:
    @pytest.fixture(
        params=[
            {'subareas': ['sub_area_1'], 'modalities': ['3K_RGB', 'osm_buildings']},
            {
                'subareas': ['sub_area_1', 'sub_area_2'],
                'modalities': ['3K_DSM', 'HySpex', 'osm_water'],
            },
            {
                'subareas': ['sub_area_2', 'sub_area_3'],
                'modalities': [
                    '3K_DSM',
                    '3K_RGB',
                    'HySpex',
                    'EeteS_EnMAP_10m',
                    'EeteS_EnMAP_30m',
                    'EeteS_Sentinel_2_10m',
                    'Sentinel_2',
                    'Sentinel_1',
                    'osm_buildings',
                    'osm_landuse',
                    'osm_water',
                ],
            },
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> MDAS:
        md5 = '99e1744ca6f19aa19a3aa23a2bbf7bef'
        monkeypatch.setattr(MDAS, 'md5', md5)
        url = os.path.join('tests', 'data', 'mdas', 'Augsburg_data_4_publication.zip')
        monkeypatch.setattr(MDAS, 'url', url)

        params = request.param
        subareas = params['subareas']
        modalities = params['modalities']

        root = tmp_path
        transforms = nn.Identity()

        return MDAS(
            root=root,
            subareas=subareas,
            modalities=modalities,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: MDAS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        for key in dataset.modalities:
            if key.startswith('osm'):
                key = f'{key}_mask'
            else:
                key = f'{key}_image'
            assert key in x

        for key, value in x.items():
            assert isinstance(value, torch.Tensor)

    def test_len(self, dataset: MDAS) -> None:
        assert len(dataset) == len(dataset.subareas)

    def test_already_downloaded(self, dataset: MDAS) -> None:
        MDAS(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        filename = 'Augsburg_data_4_publication.zip'
        dir = os.path.join('tests', 'data', 'mdas')
        shutil.copyfile(
            os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
        )
        MDAS(root=str(tmp_path))

    def test_invalid_subarea(self) -> None:
        with pytest.raises(AssertionError):
            MDAS(subareas=['foo'])

    def test_invalid_modality(self) -> None:
        with pytest.raises(AssertionError):
            MDAS(modalities=['foo'])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MDAS(tmp_path)

    def test_plot(self, dataset: MDAS) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()
