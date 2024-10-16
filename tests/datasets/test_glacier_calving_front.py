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

from torchgeo.datasets import DatasetNotFoundError, GlacierCalvingFront


class TestGlacierCalvingFront:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> GlacierCalvingFront:
        md5 = '0b5c05bea31ff666f8eba18b43d4a01f'
        monkeypatch.setattr(GlacierCalvingFront, 'md5', md5)
        url = os.path.join(
            'tests', 'data', 'glacier_calving_front', 'glacier_calving_data.zip'
        )
        monkeypatch.setattr(GlacierCalvingFront, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return GlacierCalvingFront(
            root, split, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: GlacierCalvingFront) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape[0] == 1
        assert isinstance(x['mask_zone'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask_zone'].shape[-2:]

    def test_len(self, dataset: GlacierCalvingFront) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 3

    def test_already_downloaded(self, dataset: GlacierCalvingFront) -> None:
        GlacierCalvingFront(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        filename = 'glacier_calving_data.zip'
        dir = os.path.join('tests', 'data', 'glacier_calving_front')
        shutil.copyfile(
            os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
        )
        GlacierCalvingFront(root=str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            GlacierCalvingFront(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            GlacierCalvingFront(tmp_path)

    def test_plot(self, dataset: GlacierCalvingFront) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask_zone'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()
