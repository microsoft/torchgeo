# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import DatasetNotFoundError, TropicalCyclone
from torchgeo.datasets.utils import Executable


class TestTropicalCyclone:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self,
        request: SubRequest,
        azcopy: Executable,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> TropicalCyclone:
        url = os.path.join('tests', 'data', 'cyclone')
        monkeypatch.setattr(TropicalCyclone, 'url', url)
        monkeypatch.setattr(TropicalCyclone, 'size', 2)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return TropicalCyclone(root, split, transforms, download=True)

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: TropicalCyclone, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['relative_time'], torch.Tensor)
        assert isinstance(x['ocean'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape == (3, dataset.size, dataset.size)

    def test_len(self, dataset: TropicalCyclone) -> None:
        assert len(dataset) == 5

    def test_add(self, dataset: TropicalCyclone) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 10

    def test_already_downloaded(self, dataset: TropicalCyclone) -> None:
        TropicalCyclone(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            TropicalCyclone(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            TropicalCyclone(tmp_path)

    def test_plot(self, dataset: TropicalCyclone) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
