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

import torchgeo.datasets.utils
from torchgeo.datasets import LoveDA


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLoveDA:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LoveDA:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        md5 = "3d5b1373ef9a3084ec493b9b2056fe07"

        info_dict = {
            "train": {
                "url": os.path.join("tests", "data", "loveda", "Train.zip"),
                "filename": "Train.zip",
                "md5": md5,
            },
            "val": {
                "url": os.path.join("tests", "data", "loveda", "Val.zip"),
                "filename": "Val.zip",
                "md5": md5,
            },
            "test": {
                "url": os.path.join("tests", "data", "loveda", "Test.zip"),
                "filename": "Test.zip",
                "md5": md5,
            },
        }

        monkeypatch.setattr(LoveDA, "info_dict", info_dict)

        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return LoveDA(
            root=root, split=split, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: LoveDA) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].shape[0] == 3

        if dataset.split != "test":
            assert isinstance(x["mask"], torch.Tensor)
            assert x["image"].shape[-2:] == x["mask"].shape[-2:]
        else:
            assert "mask" not in x

    def test_len(self, dataset: LoveDA) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LoveDA) -> None:
        print(dataset.root)
        LoveDA(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LoveDA(split="foo")

    def test_invalid_scene(self) -> None:
        with pytest.raises(AssertionError):
            LoveDA(scene=["garden"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(
            RuntimeError, match="Dataset not found at root directory or corrupted."
        ):
            LoveDA(str(tmp_path))

    def test_plot(self, dataset: LoveDA) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()
