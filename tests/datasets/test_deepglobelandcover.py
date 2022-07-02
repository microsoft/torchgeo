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

from torchgeo.datasets import DeepGlobeLandCover


class TestDeepGlobeLandCover:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, request: SubRequest
    ) -> DeepGlobeLandCover:
        md5 = "2cbd68d36b1485f09f32d874dde7c5c5"
        monkeypatch.setattr(DeepGlobeLandCover, "md5", md5)
        root = os.path.join("tests", "data", "deepglobelandcover")
        split = request.param
        transforms = nn.Identity()
        return DeepGlobeLandCover(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: DeepGlobeLandCover) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: DeepGlobeLandCover) -> None:
        assert len(dataset) == 3

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join("tests", "data", "deepglobelandcover")
        filename = "data.zip"
        shutil.copyfile(
            os.path.join(root, filename), os.path.join(str(tmp_path), filename)
        )
        DeepGlobeLandCover(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "data.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            DeepGlobeLandCover(root=str(tmp_path), checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            DeepGlobeLandCover(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(
            RuntimeError,
            match="Dataset not found in `root`, either"
            + " specify a different `root` directory or manually download"
            + " the dataset to this directory.",
        ):
            DeepGlobeLandCover(str(tmp_path))

    def test_plot(self, dataset: DeepGlobeLandCover) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
