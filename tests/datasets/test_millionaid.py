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

import torchgeo.datasets.utils
from torchgeo.datasets import MillionAID


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestMillionAID:
    @pytest.fixture(
        params=zip(
            ["train", "train", "test", "test"],
            ["multi-class", "multi-label", "multi-class", "multi-label"],
        )
    )
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        request: SubRequest,
        tmp_path: Path,
    ) -> MillionAID:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        data_dir = os.path.join("tests", "data", "millionaid")

        urls = {
            "train": os.path.join(data_dir, "train.zip"),
            "test": os.path.join(data_dir, "test.zip"),
        }

        md5s = {
            "train": "d5b7c0e90af70b4e6746c9d3a37471b2",
            "test": "7309f19eca7f010d1af9a6adb396b7f8",
        }

        monkeypatch.setattr(MillionAID, "url", urls)  # type: ignore[attr-defined]
        monkeypatch.setattr(MillionAID, "md5s", md5s)  # type: ignore[attr-defined]
        root = str(tmp_path)
        split, task = request.param
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return MillionAID(
            root=root,
            split=split,
            task=task,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_already_downloaded(self, dataset: MillionAID) -> None:
        MillionAID(root=dataset.root, split=dataset.split, download=True)

    def test_getitem(self, dataset: MillionAID) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["image"].ndim == 3

    def test_len(self, dataset: MillionAID) -> None:
        assert len(dataset) == 2

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join("tests", "data", "millionaid", "train.zip")
        shutil.copy(url, tmp_path)
        MillionAID(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "train.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            MillionAID(root=str(tmp_path), checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in."):
            MillionAID(str(tmp_path))

    def test_plot(self, dataset: MillionAID) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: MillionAID) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
