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
from torchgeo.datasets import LEVIRCDPlus


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLEVIRCDPlus:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LEVIRCDPlus:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        md5 = "1adf156f628aa32fb2e8fe6cada16c04"
        monkeypatch.setattr(LEVIRCDPlus, "md5", md5)
        url = os.path.join("tests", "data", "levircd", "LEVIR-CD+.zip")
        monkeypatch.setattr(LEVIRCDPlus, "url", url)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return LEVIRCDPlus(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LEVIRCDPlus) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 2

    def test_len(self, dataset: LEVIRCDPlus) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LEVIRCDPlus) -> None:
        LEVIRCDPlus(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LEVIRCDPlus(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            LEVIRCDPlus(str(tmp_path))

    def test_plot(self, dataset: LEVIRCDPlus) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="Prediction")
        plt.close()
