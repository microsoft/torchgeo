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

import torchgeo
from torchgeo.datasets import DatasetNotFoundError, DigitalTyphoonAnalysis

pytest.importorskip('h5py', minversion='3.6')


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestDigitalTyphoon:
    @pytest.fixture(
        params=[
            (3, {'wind': 0}, {'pressure': 1500}),
            (3, {'pressure': 0}, {'wind': 100}),
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DigitalTyphoonAnalysis:
        sequence_length, min_features, max_features = request.param
        monkeypatch.setattr(
            torchgeo.datasets.digital_typhoon, 'download_url', download_url
        )

        url = os.path.join('tests', 'data', 'digital_typhoon', 'WP.tar.gz{0}')
        monkeypatch.setattr(DigitalTyphoonAnalysis, 'url', url)

        md5sums = {
            'aa': '5b2fed45d9719e77a482ccd4ae1b02e5',
            'ab': '5b2fed45d9719e77a482ccd4ae1b02e5',
        }
        monkeypatch.setattr(DigitalTyphoonAnalysis, 'md5sums', md5sums)
        root = str(tmp_path)

        transforms = nn.Identity()
        return DigitalTyphoonAnalysis(
            root=root,
            sequence_length=sequence_length,
            min_feature_value=min_features,
            max_feature_value=max_features,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_len(self, dataset: DigitalTyphoonAnalysis) -> None:
        assert len(dataset) == 10

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: DigitalTyphoonAnalysis, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].min() >= 0 and x['image'].max() <= 1
        assert isinstance(x['label'], torch.Tensor)

    def test_already_downloaded(self, dataset: DigitalTyphoonAnalysis) -> None:
        DigitalTyphoonAnalysis(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'digital_typhoon')
        filenames = ['WP.tar.gzaa', 'WP.tar.gzab']
        for filename in filenames:
            shutil.copyfile(
                os.path.join(root, filename), os.path.join(str(tmp_path), filename)
            )
        DigitalTyphoonAnalysis(root=str(tmp_path))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DigitalTyphoonAnalysis(root=str(tmp_path))

    def test_plot(self, dataset: DigitalTyphoonAnalysis) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
