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

from torchgeo.datasets import DatasetNotFoundError, DynamicEarthNet


class TestDynamicEarthNet:
    @pytest.fixture(
        params=product(['train', 'val', 'test'], ['monthly', 'weekly', 'daily'])
    )
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> DynamicEarthNet:
        filename_and_md5: ClassVar[dict[str, dict[str, str]]] = {
            'planet': {
                'filename': 'planet_pf_sr.tar.gz',
                'md5': 'd41d8cd98f00b204e9800998ecf8427e',
            },
            # 's1': {
            #     'filename': 's1.tar.gz',
            #     'md5': 'd41d8cd98f00b204e9800998ecf8427e',
            # },
            # 's2': {
            #     'filename': 's2.tar.gz',
            #     'md5': 'd41d8cd98f00b204e9800998ecf8427e',
            # },
            'labels': {
                'filename': 'labels.tar.gz',
                'md5': 'd41d8cd98f00b204e9800998ecf8427e',
            },
            'split_info': {
                'filename': 'split_info.tar.gz',
                'md5': 'd41d8cd98f00b204e9800998ecf8427e',
            },
        }
        monkeypatch.setattr(DynamicEarthNet, 'filename_and_md5', filename_and_md5)
        root = os.path.join('tests', 'data', 'dynamic_earthnet')
        split, tempooral_input = request.param
        transforms = nn.Identity()
        return DynamicEarthNet(
            root,
            split,
            temporal_input=temporal_input,
            transforms=transforms,
            checksum=True,
            download=True,
        )

    def test_getitem(self, dataset: DynamicEarthNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

        if dataset.temporal_inputs == 'monthly':
            assert x['image'].shape[0] == 1
        elif dataset.temporal_inputs == 'weekly':
            assert x['image'].shape[0] == 6
        elif dataset.temporal_inputs == 'daily':
            assert x['image'].shape[0] >= 28

    def test_additional_modality(self, dataset: DynamicEarthNet) -> None:
        x = dataset[0]
        if 's1' in dataset.add_modalities:
            assert isinstance(x['s1_image'], torch.Tensor)
            assert x['s1_image'].shape[0] == 1
        if 's2' in dataset.add_modalities:
            assert isinstance(x['s2_image'], torch.Tensor)
            assert x['s2_image'].shape[0] == 13

    def test_len(self, dataset: DynamicEarthNet) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'dynamic_earthnet')
        filenames = [
            'planet_pf_sr.tar.gz',
            'sentinel1.tar.gz',
            'sentinel2.tar.gz',
            'split_info.tar.gz',
            'labels.tar.gz',
        ]
        for filename in filenames:
            shutil.copyfile(
                os.path.join(root, filename), os.path.join(tmp_path, filename)
            )
        DynamicEarthNet(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'labels.tar.gz'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            DynamicEarthNet(root=tmp_path, checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            DynamicEarthNet(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DynamicEarthNet(tmp_path)

    def test_plot(self, dataset: DynamicEarthNet) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
