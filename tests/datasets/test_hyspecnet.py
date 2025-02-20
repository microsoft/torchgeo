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

from torchgeo.datasets import DatasetNotFoundError, HySpecNet11k, RGBBandsMissingError

root = os.path.join('tests', 'data', 'hyspecnet')
md5s = {'hyspecnet-11k-01.tar.gz': '', 'hyspecnet-11k-splits.tar.gz': ''}


class TestHySpecNet11k:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch) -> HySpecNet11k:
        monkeypatch.setattr(HySpecNet11k, 'url', root + os.sep)
        monkeypatch.setattr(HySpecNet11k, 'md5s', md5s)
        transforms = nn.Identity()
        return HySpecNet11k(root, transforms=transforms)

    def test_getitem(self, dataset: HySpecNet11k) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], Tensor)

    def test_len(self, dataset: HySpecNet11k) -> None:
        assert len(dataset) == 2

    def test_download(self, dataset: HySpecNet11k, tmp_path: Path) -> None:
        HySpecNet11k(tmp_path, download=True)

    def test_extract(self, dataset: HySpecNet11k, tmp_path: Path) -> None:
        for file in glob.iglob(os.path.join(root, '*.tar.gz')):
            shutil.copy(file, tmp_path)
        HySpecNet11k(tmp_path)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            HySpecNet11k(tmp_path)

    def test_plot(self, dataset: HySpecNet11k) -> None:
        x = dataset[0]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_rgb(self, dataset: HySpecNet11k) -> None:
        dataset = HySpecNet11k(root=dataset.root, bands=('B1', 'B2', 'B3'))
        match = 'Dataset does not contain some of the RGB bands'
        with pytest.raises(RGBBandsMissingError, match=match):
            dataset.plot(dataset[0])
