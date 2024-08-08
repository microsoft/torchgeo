# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SpaceNet1
from torchgeo.datasets.utils import Executable


class TestSpaceNet:
    @pytest.fixture
    def dataset(
        self, aws: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet1:
        url = os.path.join(
            'tests', 'data', 'spacenet', '{dataset_id}', 'tarballs', '{tarball}'
        )
        monkeypatch.setattr(SpaceNet1, 'url', url)
        transforms = nn.Identity()
        return SpaceNet1(tmp_path, transforms=transforms, download=True)

    def test_getitem(self, dataset: SpaceNet1) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SpaceNet1) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: SpaceNet1) -> None:
        SpaceNet1(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SpaceNet1(tmp_path)

    def test_plot(self, dataset: SpaceNet1) -> None:
        x = dataset[0].copy()
        x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
