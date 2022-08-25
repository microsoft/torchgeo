# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import PatternNet


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestPatternNet:
    @pytest.fixture(params=["train", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> PatternNet:
        monkeypatch.setattr(torchgeo.datasets.patternnet, "download_url", download_url)
        md5 = "5649754c78219a2c19074ff93666cc61"
        monkeypatch.setattr(PatternNet, "md5", md5)
        url = os.path.join("tests", "data", "patternnet", "PatternNet.zip")
        monkeypatch.setattr(PatternNet, "url", url)
        root = str(tmp_path)
        transforms = nn.Identity()
        return PatternNet(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: PatternNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3

    def test_len(self, dataset: PatternNet) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: PatternNet, tmp_path: Path) -> None:
        PatternNet(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: PatternNet, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        PatternNet(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            PatternNet(str(tmp_path))

    def test_plot(self, dataset: PatternNet) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["label"].clone()
        dataset.plot(sample, suptitle="Prediction")
        plt.close()
