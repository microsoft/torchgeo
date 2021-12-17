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
from torchgeo.datasets import COWCCounting, COWCCountingDataModule, COWCDetection
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
            "a729b6e29278a9a000aa349dad3c78cb",
            "a8ff4c4de4b8c66bd9c5ec17f532b3a2",
            "bc6b9493b8e39b87d189cadcc4823e05",
            "f111948e2ac262c024c8fe32ba5b1434",
            "8c333fcfa4168afa5376310958d15166",
            "479670049aa9a48b4895cff6db3aa615",
            "56043d4716ad0a1eedd392b0a543973b",
            "b77193aef7c473379cd8d4e40d413137",
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
            "cc913824d9aa6c7af6f957dcc2cb9690",
            "f8e07e70958d8d57ab464f62e9abab80",
            "6a481cd785b0f16e9e1ab016a0695e57",
            "e9578491977d291def2611b84c84fdfd",
            "0bb1c285b170c23a8590cf2926fd224e",
            "60fa485b16c0e5b28db756fd1d8a0438",
            "97c886fb7558f4e8779628917ca64596",
            "ab21a117b754e04e65c63f94aa648e33",
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


class TestCOWCCountingDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> COWCCountingDataModule:
        root = os.path.join("tests", "data", "cowc_counting")
        seed = 0
        batch_size = 1
        num_workers = 0
        dm = COWCCountingDataModule(root, seed, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: COWCCountingDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: COWCCountingDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: COWCCountingDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
