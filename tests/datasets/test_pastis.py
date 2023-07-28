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
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import (
    PASTIS,
    PASTISInstanceSegmentation,
    PASTISSemanticSegmentation,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestPASTIS:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> PASTIS:
        monkeypatch.setattr(torchgeo.datasets.pastis, "download_url", download_url)

        md5 = "9b11ae132623a0d13f7f0775d2003703"
        monkeypatch.setattr(PASTIS, "md5", md5)
        url = os.path.join("tests", "data", "pastis", "PASTIS-R.zip")
        monkeypatch.setattr(PASTIS, "url", url)
        root = str(tmp_path)
        transforms = nn.Identity()
        return PASTIS(root, (0, 1), "s2", transforms, download=True, checksum=True)

    def test_getitem_not_implemented(self, dataset: PASTIS) -> None:
        with pytest.raises(NotImplementedError):
            dataset[0]

    def test_load_target_not_implemented(self, dataset: PASTIS) -> None:
        with pytest.raises(NotImplementedError):
            dataset._load_target(0)


class TestPASTISSemanticSegmentation:
    @pytest.fixture(
        params=[
            {"folds": (0, 1), "bands": "s2"},
            {"folds": (0, 1), "bands": "s1a"},
            {"folds": (0, 1), "bands": "s1d"},
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> PASTISSemanticSegmentation:
        monkeypatch.setattr(torchgeo.datasets.pastis, "download_url", download_url)

        md5 = "9b11ae132623a0d13f7f0775d2003703"
        monkeypatch.setattr(PASTIS, "md5", md5)
        url = os.path.join("tests", "data", "pastis", "PASTIS-R.zip")
        monkeypatch.setattr(PASTIS, "url", url)
        root = str(tmp_path)
        folds = request.param["folds"]
        bands = request.param["bands"]
        transforms = nn.Identity()
        return PASTISSemanticSegmentation(
            root, folds, bands, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: PASTISSemanticSegmentation) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: PASTISSemanticSegmentation) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: PASTISSemanticSegmentation) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: PASTISSemanticSegmentation) -> None:
        PASTISSemanticSegmentation(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        url = os.path.join("tests", "data", "pastis", "PASTIS-R.zip")
        root = str(tmp_path)
        shutil.copy(url, root)
        PASTISSemanticSegmentation(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            PASTISSemanticSegmentation(str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "PASTIS-R.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            PASTISSemanticSegmentation(root=str(tmp_path), checksum=True)

    def test_invalid_fold(self) -> None:
        with pytest.raises(AssertionError):
            PASTISSemanticSegmentation(folds=(6,))

    def test_plot(self, dataset: PASTISSemanticSegmentation) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()


class TestPASTISInstanceSegmentation:
    @pytest.fixture(
        params=[
            {"folds": (0, 1), "bands": "s2"},
            {"folds": (0, 1), "bands": "s1a"},
            {"folds": (0, 1), "bands": "s1d"},
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> PASTISInstanceSegmentation:
        monkeypatch.setattr(torchgeo.datasets.pastis, "download_url", download_url)

        md5 = "9b11ae132623a0d13f7f0775d2003703"
        monkeypatch.setattr(PASTIS, "md5", md5)
        url = os.path.join("tests", "data", "pastis", "PASTIS-R.zip")
        monkeypatch.setattr(PASTIS, "url", url)
        root = str(tmp_path)
        folds = request.param["folds"]
        bands = request.param["bands"]
        transforms = nn.Identity()
        return PASTISInstanceSegmentation(
            root, folds, bands, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: PASTISSemanticSegmentation) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["boxes"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)