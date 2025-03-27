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

from torchgeo.datasets import DatasetNotFoundError, DigitalTyphoon

pytest.importorskip('h5py', minversion='3.8')


class TestDigitalTyphoon:
    @pytest.fixture(
        params=[
            (3, {'wind': 0}, {'pressure': 1500}),
            (3, {'pressure': 0}, {'wind': 100}),
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DigitalTyphoon:
        sequence_length, min_features, max_features = request.param

        url = os.path.join('tests', 'data', 'digital_typhoon', 'WP.tar.gz{0}')
        monkeypatch.setattr(DigitalTyphoon, 'url', url)

        md5sums = {
            'aa': '692ea3796c9bc9ef1e0ab6f2b8bc51ad',
            'ab': '692ea3796c9bc9ef1e0ab6f2b8bc51ad',
        }
        monkeypatch.setattr(DigitalTyphoon, 'md5sums', md5sums)
        root = tmp_path

        transforms = nn.Identity()
        return DigitalTyphoon(
            root=root,
            sequence_length=sequence_length,
            min_feature_value=min_features,
            max_feature_value=max_features,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_len(self, dataset: DigitalTyphoon) -> None:
        assert len(dataset) == 15

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: DigitalTyphoon, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].min() >= 0 and x['image'].max() <= 1
        assert isinstance(x['label'], torch.Tensor)

    def test_already_downloaded(self, dataset: DigitalTyphoon) -> None:
        DigitalTyphoon(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'digital_typhoon')
        filenames = ['WP.tar.gzaa', 'WP.tar.gzab']
        for filename in filenames:
            shutil.copyfile(os.path.join(root, filename), tmp_path / filename)
        DigitalTyphoon(root=str(tmp_path))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DigitalTyphoon(root=str(tmp_path))

    def test_plot(self, dataset: DigitalTyphoon) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
