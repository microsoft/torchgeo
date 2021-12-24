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
from torchgeo.datasets import LandCoverAI


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLandCoverAI:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> LandCoverAI:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "5bf4d2770deb41eb5c38784ab2c8a691"
        monkeypatch.setattr(LandCoverAI, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip")
        monkeypatch.setattr(LandCoverAI, "url", url)  # type: ignore[attr-defined]
        sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        monkeypatch.setattr(LandCoverAI, "sha256", sha256)  # type: ignore[attr-defined]
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return LandCoverAI(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LandCoverAI) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: LandCoverAI) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: LandCoverAI) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: LandCoverAI) -> None:
        LandCoverAI(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LandCoverAI(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            LandCoverAI(str(tmp_path))

    def test_plot(self, dataset: LandCoverAI) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
