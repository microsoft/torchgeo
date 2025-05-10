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

from torchgeo.datasets import DatasetNotFoundError, PatternNet


class TestPatternNet:
    @pytest.fixture(params=['train', 'test'])
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> PatternNet:
        md5 = '5649754c78219a2c19074ff93666cc61'
        monkeypatch.setattr(PatternNet, 'md5', md5)
        url = os.path.join('tests', 'data', 'patternnet', 'PatternNet.zip')
        monkeypatch.setattr(PatternNet, 'url', url)
        root = tmp_path
        transforms = nn.Identity()
        return PatternNet(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: PatternNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape[0] == 3

    def test_len(self, dataset: PatternNet) -> None:
        assert len(dataset) == 6

    def test_already_downloaded(self, dataset: PatternNet, tmp_path: Path) -> None:
        PatternNet(root=tmp_path, download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: PatternNet, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        shutil.copy(dataset.url, tmp_path)
        PatternNet(root=tmp_path, download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            PatternNet(tmp_path)

    def test_plot(self, dataset: PatternNet) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['label'].clone()
        dataset.plot(sample, suptitle='Prediction')
        plt.close()
