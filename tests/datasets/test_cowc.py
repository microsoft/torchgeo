# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import COWCCounting, COWCDetection
from torchgeo.datasets.cowc import COWC


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestCOWC:
    def test_not_implemented(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            COWC()  # type: ignore[abstract]


class TestCOWCCounting:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> COWC:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        base_url = os.path.join("tests", "data", "cowc_counting") + os.sep
        monkeypatch.setattr(  # type: ignore[attr-defined]
            COWCCounting, "base_url", base_url
        )
        md5s = [
            "660e70330f6f14d06fe21bb6c18456d3",
            "029099f1daff5402c235c5ecf6f4bfca",
            "652a6f13242475b92312be681a2da120",
            "8c14d5ba1248ebd94f877f58299e5864",
            "1566d8a1255c22bdd3b12e901081bf0d",
            "7455ce50757674c1d0351901a575b0d0",
            "33eb18145fdb76d35b0d25a46f51500d",
            "c6da2ae0694aab9e85be4d2a60b68936",
        ]
        monkeypatch.setattr(COWCCounting, "md5s", md5s)  # type: ignore[attr-defined]
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return COWCCounting(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: COWC) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

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

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            COWCCounting(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            COWCCounting(str(tmp_path))

    def test_plot(self, dataset: COWCCounting) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()


class TestCOWCDetection:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> COWC:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        base_url = os.path.join("tests", "data", "cowc_detection") + os.sep
        monkeypatch.setattr(  # type: ignore[attr-defined]
            COWCDetection, "base_url", base_url
        )
        md5s = [
            "8f74bb77c7e3032e451dae2ea5809c7b",
            "253b9bd4da1cfedd4e5ce87751710e97",
            "e272b0d36ac747b9cb5e3977be4cedf9",
            "5cbff32b00213a5403a80b46d65acda2",
            "d57ae3c60d6746a41861c4be03b8ed72",
            "7d656906eedb9c5cda716ec6438ba549",
            "44b801efa5c394b981c60ebe7f011465",
            "6acfad4a31a914ff3b6e4af27406aae3",
        ]
        monkeypatch.setattr(COWCDetection, "md5s", md5s)  # type: ignore[attr-defined]
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return COWCDetection(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: COWC) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

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

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            COWCDetection(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            COWCDetection(str(tmp_path))

    def test_plot(self, dataset: COWCDetection) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
