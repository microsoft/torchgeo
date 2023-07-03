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

from torchgeo.datasets import AgriFieldNet


class TestAgriFieldNet:
    @pytest.fixture(params=["train", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> AgriFieldNet:
        md5 = "aa39edc40b37d2deab4115d8c2ffeced"
        monkeypatch.setattr(AgriFieldNet, "md5", md5)
        root = os.path.join("tests", "data", "agrifieldnet")
        split = request.param
        transforms = nn.Identity()
        return AgriFieldNet(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: AgriFieldNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: AgriFieldNet) -> None:
        assert len(dataset) == 1165

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join("tests", "data", "agrifieldnet")
        filename = "ref_agrifieldnet_competition_v1.tar.gz"
        shutil.copyfile(
            os.path.join(root, filename), os.path.join(str(tmp_path), filename)
        )
        AgriFieldNet(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(
            os.path.join(tmp_path, "ref_agrifieldnet_competition_v1.tar.gz"), "w"
        ) as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            AgriFieldNet(root=str(tmp_path), checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            AgriFieldNet(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(
            RuntimeError,
            match="Dataset not found in `root`, either"
            + " specify a different `root` directory or manually download"
            + " the dataset to this directory.",
        ):
            AgriFieldNet(str(tmp_path))

    def test_plot(self, dataset: AgriFieldNet) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
