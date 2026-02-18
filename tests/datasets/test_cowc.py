# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import COWC, COWCCounting, COWCDetection, DatasetNotFoundError


class TestCOWC:
    def test_not_implemented(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            COWC()


class TestCOWCCounting:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> COWC:
        base_url = os.path.join('tests', 'data', 'cowc_counting') + os.sep
        monkeypatch.setattr(COWCCounting, 'base_url', base_url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return COWCCounting(root, split, transforms, download=True)

    def test_getitem(self, dataset: COWC) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_len(self, dataset: COWC) -> None:
        assert len(dataset) in [6, 12]

    def test_add(self, dataset: COWC) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) in [12, 24]

    def test_already_downloaded(self, dataset: COWC) -> None:
        COWCCounting(root=dataset.root, download=True)

    def test_out_of_bounds(self, dataset: COWC) -> None:
        with pytest.raises(IndexError):
            dataset[12]

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            COWCCounting(tmp_path)

    def test_plot(self, dataset: COWCCounting) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()


class TestCOWCDetection:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> COWC:
        base_url = os.path.join('tests', 'data', 'cowc_detection') + os.sep
        monkeypatch.setattr(COWCDetection, 'base_url', base_url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return COWCDetection(root, split, transforms, download=True)

    def test_getitem(self, dataset: COWC) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_len(self, dataset: COWC) -> None:
        assert len(dataset) in [6, 12]

    def test_add(self, dataset: COWC) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) in [12, 24]

    def test_already_downloaded(self, dataset: COWC) -> None:
        COWCDetection(root=dataset.root, download=True)

    def test_out_of_bounds(self, dataset: COWC) -> None:
        with pytest.raises(IndexError):
            dataset[12]

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            COWCDetection(tmp_path)

    def test_plot(self, dataset: COWCDetection) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
