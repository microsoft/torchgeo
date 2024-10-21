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

from torchgeo.datasets import CaFFe, DatasetNotFoundError


class TestCaFFe:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> CaFFe:
        md5 = 'f06c155a3fea372e884c234115c169e1'
        monkeypatch.setattr(CaFFe, 'md5', md5)
        url = os.path.join('tests', 'data', 'caffe', 'caffe.zip')
        monkeypatch.setattr(CaFFe, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return CaFFe(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: CaFFe) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape[0] == 1
        assert isinstance(x['mask_zones'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask_zones'].shape[-2:]

    def test_len(self, dataset: CaFFe) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 3

    def test_already_downloaded(self, dataset: CaFFe) -> None:
        CaFFe(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        filename = 'caffe.zip'
        dir = os.path.join('tests', 'data', 'caffe')
        shutil.copyfile(
            os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
        )
        CaFFe(root=str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            CaFFe(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CaFFe(tmp_path)

    def test_plot(self, dataset: CaFFe) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask_zones'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()
