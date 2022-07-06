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
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> COWC:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        base_url = os.path.join("tests", "data", "cowc_counting") + os.sep
        monkeypatch.setattr(COWCCounting, "base_url", base_url)
        md5s = [
            "7d0c6d1fb548d3ea3a182a56ce231f97",
            "2e9a806b19b21f9d796c7393ad8f51ee",
            "39453c0627effd908e773c5c1f8aecc9",
            "67190b3e0ca8aa1fc93250aa5383a8f3",
            "575aead6a0c92aba37d613895194da7c",
            "e7c2279040d3ce31b9c925c45d0c61e2",
            "f159e23d52bd0b5656fe296f427b98e1",
            "0a4daed8c5f6c4e20faa6e38636e4346",
        ]
        monkeypatch.setattr(COWCCounting, "md5s", md5s)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
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
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> COWC:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        base_url = os.path.join("tests", "data", "cowc_detection") + os.sep
        monkeypatch.setattr(COWCDetection, "base_url", base_url)
        md5s = [
            "6bbbdb36ee4922e879f66ed9234cb8ab",
            "09e4af08c6e6553afe5098b328ce9749",
            "12a2708ab7644766e43f5aae34aa7f2a",
            "a896433398a0c58263c0d266cfc93bc4",
            "911ed42c104db60f7a7d03a5b36bc1ab",
            "4cdb4fefab6a2951591e7840c11a229d",
            "dd315cfb48dfa7ddb8230c942682bc37",
            "dccc2257e9c4a9dde2b4f84769804046",
        ]
        monkeypatch.setattr(COWCDetection, "md5s", md5s)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
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
