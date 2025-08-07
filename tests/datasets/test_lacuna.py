# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import DatasetNotFoundError, LacunaAfricanFieldBoundaries


class TestLacunaAfricanFieldBoundaries:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> LacunaAfricanFieldBoundaries:
        monkeypatch.setattr(
            LacunaAfricanFieldBoundaries, 'md5', '2fda01ab513b574ae97fbd92168189c7'
        )
        url = os.path.join('tests', 'data', 'lacuna', 'lacuna-field-boundaries.tar.gz')
        monkeypatch.setattr(LacunaAfricanFieldBoundaries, 'url', url)
        root = tmp_path
        transforms = nn.Identity()
        return LacunaAfricanFieldBoundaries(
            root, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: LacunaAfricanFieldBoundaries) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: LacunaAfricanFieldBoundaries) -> None:
        assert len(dataset) == 8

    def test_add(self, dataset: LacunaAfricanFieldBoundaries) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 16

    def test_already_extracted(self, dataset: LacunaAfricanFieldBoundaries) -> None:
        LacunaAfricanFieldBoundaries(root=dataset.root, download=True)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'lacuna', 'lacuna-field-boundaries.tar.gz')
        root = tmp_path
        shutil.copy(url, root)
        LacunaAfricanFieldBoundaries(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            LacunaAfricanFieldBoundaries(tmp_path)

    def test_plot(self, dataset: LacunaAfricanFieldBoundaries) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
