# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    DatasetNotFoundError,
    LandCoverAI,
    LandCoverAI100,
    LandCoverAIGeo,
)


class TestLandCoverAIGeo:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> LandCoverAIGeo:
        url = os.path.join('tests', 'data', 'landcoverai', 'landcover.ai.v1.zip')
        monkeypatch.setattr(LandCoverAIGeo, 'url', url)
        root = tmp_path
        transforms = nn.Identity()
        return LandCoverAIGeo(root, transforms=transforms, download=True)

    def test_getitem(self, dataset: LandCoverAIGeo) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_already_extracted(self, dataset: LandCoverAIGeo) -> None:
        LandCoverAIGeo(dataset.root, download=True)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'landcoverai', 'landcover.ai.v1.zip')
        root = tmp_path
        shutil.copy(url, root)
        LandCoverAIGeo(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            LandCoverAIGeo(tmp_path)

    def test_out_of_bounds_index(self, dataset: LandCoverAIGeo) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_plot(self, dataset: LandCoverAIGeo) -> None:
        x = dataset[dataset.bounds].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'][:, :, 0].clone().unsqueeze(2)
        dataset.plot(x)
        plt.close()


class TestLandCoverAI:
    @pytest.fixture(
        params=product([LandCoverAI100, LandCoverAI], ['train', 'val', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LandCoverAI:
        base_class: type[LandCoverAI] = request.param[0]
        split: str = request.param[1]
        url = os.path.join('tests', 'data', 'landcoverai', 'landcover.ai.v1.zip')
        monkeypatch.setattr(base_class, 'url', url)
        monkeypatch.setattr(base_class, 'filename', 'landcover.ai.v1.zip')
        root = tmp_path
        transforms = nn.Identity()
        return base_class(root, split, transforms, download=True)

    def test_getitem(self, dataset: LandCoverAI) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: LandCoverAI) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: LandCoverAI) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: LandCoverAI) -> None:
        LandCoverAI(root=dataset.root, download=True)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'landcoverai', 'landcover.ai.v1.zip')
        root = tmp_path
        # Copy with the expected filename for LandCoverAI
        shutil.copy(url, os.path.join(root, 'output.zip'))
        LandCoverAI(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            LandCoverAI(tmp_path)

    def test_plot(self, dataset: LandCoverAI) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
