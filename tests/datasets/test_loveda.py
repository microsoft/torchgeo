# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, LoveDA


class TestLoveDA:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LoveDA:
        info_dict = {
            'train': {
                'url': os.path.join('tests', 'data', 'loveda', 'Train.zip'),
                'filename': 'Train.zip',
                'md5': '',
            },
            'val': {
                'url': os.path.join('tests', 'data', 'loveda', 'Val.zip'),
                'filename': 'Val.zip',
                'md5': '',
            },
            'test': {
                'url': os.path.join('tests', 'data', 'loveda', 'Test.zip'),
                'filename': 'Test.zip',
                'md5': '',
            },
        }

        monkeypatch.setattr(LoveDA, 'info_dict', info_dict)

        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return LoveDA(root=root, split=split, transforms=transforms, download=True)

    def test_getitem(self, dataset: LoveDA) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape[0] == 3

        if dataset.split != 'test':
            assert isinstance(x['mask'], torch.Tensor)
            assert x['image'].shape[-2:] == x['mask'].shape[-2:]
        else:
            assert 'mask' not in x

    def test_len(self, dataset: LoveDA) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LoveDA) -> None:
        print(dataset.root)
        LoveDA(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            LoveDA(tmp_path)

    def test_plot(self, dataset: LoveDA) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()
