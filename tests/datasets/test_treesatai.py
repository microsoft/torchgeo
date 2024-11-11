# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch.nn as nn
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.datasets import DatasetNotFoundError, TreeSatAI

root = os.path.join('tests', 'data', 'treesatai')
md5s = {
    'aerial_60m_acer_pseudoplatanus.zip': '',
    'labels.zip': '',
    's1.zip': '',
    's2.zip': '',
    'test_filenames.lst': '',
    'train_filenames.lst': '',
}


class TestTreeSatAI:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch) -> TreeSatAI:
        monkeypatch.setattr(TreeSatAI, 'url', root + os.sep)
        monkeypatch.setattr(TreeSatAI, 'md5s', md5s)
        transforms = nn.Identity()
        return TreeSatAI(root, transforms=transforms)

    def test_getitem(self, dataset: TreeSatAI) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['label'], Tensor)
        for sensor in dataset.sensors:
            assert isinstance(x[f'image_{sensor}'], Tensor)

    def test_len(self, dataset: TreeSatAI) -> None:
        assert len(dataset) == 9

    def test_download(self, dataset: TreeSatAI, tmp_path: Path) -> None:
        TreeSatAI(tmp_path, download=True)

    def test_extract(self, dataset: TreeSatAI, tmp_path: Path) -> None:
        for file in glob.iglob(os.path.join(root, '*.*')):
            shutil.copy(file, tmp_path)
        TreeSatAI(tmp_path)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            TreeSatAI(tmp_path)

    def test_plot(self, dataset: TreeSatAI) -> None:
        x = dataset[0]
        x['prediction'] = x['label']
        dataset.plot(x)
        plt.close()
