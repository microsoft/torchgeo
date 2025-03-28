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

from torchgeo.datasets import DatasetNotFoundError, ForestDamage


class TestForestDamage:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ForestDamage:
        data_dir = os.path.join('tests', 'data', 'forestdamage')
        url = os.path.join(data_dir, 'Data_Set_Larch_Casebearer.zip')
        md5 = '52d82ac38899e6e6bb40aacda643ee15'
        monkeypatch.setattr(ForestDamage, 'url', url)
        monkeypatch.setattr(ForestDamage, 'md5', md5)
        root = tmp_path
        transforms = nn.Identity()
        return ForestDamage(
            root=root, transforms=transforms, download=True, checksum=True
        )

    def test_already_downloaded(self, dataset: ForestDamage) -> None:
        ForestDamage(root=dataset.root, download=True)

    def test_getitem(self, dataset: ForestDamage) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3

    def test_len(self, dataset: ForestDamage) -> None:
        assert len(dataset) == 2

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join(
            'tests', 'data', 'forestdamage', 'Data_Set_Larch_Casebearer.zip'
        )
        shutil.copy(url, tmp_path)
        ForestDamage(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'Data_Set_Larch_Casebearer.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            ForestDamage(root=tmp_path, checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ForestDamage(tmp_path)

    def test_plot(self, dataset: ForestDamage) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: ForestDamage) -> None:
        x = dataset[0].copy()
        x['prediction_bbox_xyxy'] = x['bbox_xyxy'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()
