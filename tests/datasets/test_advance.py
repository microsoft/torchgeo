# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import ADVANCE, DatasetNotFoundError

pytest.importorskip('scipy', minversion='1.7.2')


class TestADVANCE:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ADVANCE:
        data_dir = os.path.join('tests', 'data', 'advance')
        urls = [
            os.path.join(data_dir, 'ADVANCE_vision.zip'),
            os.path.join(data_dir, 'ADVANCE_sound.zip'),
        ]
        md5s = ['43acacecebecd17a82bc2c1e719fd7e4', '039b7baa47879a8a4e32b9dd8287f6ad']
        monkeypatch.setattr(ADVANCE, 'urls', urls)
        monkeypatch.setattr(ADVANCE, 'md5s', md5s)
        root = tmp_path
        transforms = nn.Identity()
        return ADVANCE(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: ADVANCE) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['audio'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3
        assert x['audio'].shape[0] == 1
        assert x['audio'].ndim == 2
        assert x['label'].ndim == 0

    def test_len(self, dataset: ADVANCE) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: ADVANCE) -> None:
        ADVANCE(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ADVANCE(tmp_path)

    def test_plot(self, dataset: ADVANCE) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
