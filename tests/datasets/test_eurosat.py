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
from torchgeo.datasets import EuroSAT


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestEuroSAT:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> EuroSAT:
        monkeypatch.setattr(torchgeo.datasets.eurosat, "download_url", download_url)
        md5 = "aa051207b0547daba0ac6af57808d68e"
        monkeypatch.setattr(EuroSAT, "md5", md5)
        url = os.path.join("tests", "data", "eurosat", "EuroSATallBands.zip")
        monkeypatch.setattr(EuroSAT, "url", url)
        monkeypatch.setattr(
            EuroSAT,
            "split_urls",
            {
                "train": os.path.join("tests", "data", "eurosat", "eurosat-train.txt"),
                "val": os.path.join("tests", "data", "eurosat", "eurosat-val.txt"),
                "test": os.path.join("tests", "data", "eurosat", "eurosat-test.txt"),
            },
        )
        monkeypatch.setattr(
            EuroSAT,
            "split_md5s",
            {
                "train": "4af60a00fdfdf8500572ae5360694b71",
                "val": "4af60a00fdfdf8500572ae5360694b71",
                "test": "4af60a00fdfdf8500572ae5360694b71",
            },
        )
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return EuroSAT(
            root=root, split=split, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: EuroSAT) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            EuroSAT(split="foo")

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            EuroSAT(bands=("OK", "BK"))

    def test_len(self, dataset: EuroSAT) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: EuroSAT) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: EuroSAT, tmp_path: Path) -> None:
        EuroSAT(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: EuroSAT, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        EuroSAT(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            EuroSAT(str(tmp_path))

    def test_plot(self, dataset: EuroSAT) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_rgb(self, dataset: EuroSAT, tmp_path: Path) -> None:
        dataset = EuroSAT(root=str(tmp_path), bands=("B03",))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle="Single Band")
