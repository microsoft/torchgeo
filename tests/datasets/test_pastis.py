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
from torchgeo.datasets import PASTIS, DatasetNotFoundError


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestPASTIS:
    @pytest.fixture(
        params=[
            {"folds": (0, 1), "bands": "s2", "mode": "semantic"},
            {"folds": (0, 1), "bands": "s1a", "mode": "semantic"},
            {"folds": (0, 1), "bands": "s1d", "mode": "instance"},
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> PASTIS:
        monkeypatch.setattr(torchgeo.datasets.pastis, "download_url", download_url)

        md5 = "9b11ae132623a0d13f7f0775d2003703"
        monkeypatch.setattr(PASTIS, "md5", md5)
        url = os.path.join("tests", "data", "pastis", "PASTIS-R.zip")
        monkeypatch.setattr(PASTIS, "url", url)
        root = str(tmp_path)
        folds = request.param["folds"]
        bands = request.param["bands"]
        mode = request.param["mode"]
        transforms = nn.Identity()
        return PASTIS(
            root, folds, bands, mode, transforms, download=True, checksum=True
        )

    def test_getitem_semantic(self, dataset: PASTIS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_getitem_instance(self, dataset: PASTIS) -> None:
        dataset.mode = "instance"
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["boxes"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: PASTIS) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: PASTIS) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: PASTIS) -> None:
        PASTIS(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        url = os.path.join("tests", "data", "pastis", "PASTIS-R.zip")
        root = str(tmp_path)
        shutil.copy(url, root)
        PASTIS(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            PASTIS(str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "PASTIS-R.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            PASTIS(root=str(tmp_path), checksum=True)

    def test_invalid_fold(self) -> None:
        with pytest.raises(AssertionError):
            PASTIS(folds=(6,))

    def test_invalid_mode(self) -> None:
        with pytest.raises(AssertionError):
            PASTIS(mode="invalid")

    def test_plot(self, dataset: PASTIS) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        if dataset.mode == "instance":
            x["prediction_labels"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
