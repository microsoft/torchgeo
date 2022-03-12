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


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestLandCoverAI:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LandCoverAI:
        monkeypatch.setattr(torchgeo.datasets.landcoverai, "download_url", download_url)
        md5 = "46108372402292213789342d58929708"
        monkeypatch.setattr(LandCoverAI, "md5", md5)
        url = os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip")
        monkeypatch.setattr(LandCoverAI, "url", url)
        sha256 = "ce84fa0e8d89b461c66fba4e78aa5a860e2871722c4a9ca8c2384eae1521c7c8"
        monkeypatch.setattr(LandCoverAI, "sha256", sha256)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
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

    def test_already_extracted(self, dataset: LandCoverAI) -> None:
        LandCoverAI(root=dataset.root, download=True)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        sha256 = "ce84fa0e8d89b461c66fba4e78aa5a860e2871722c4a9ca8c2384eae1521c7c8"
        monkeypatch.setattr(LandCoverAI, "sha256", sha256)
        url = os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip")
        root = str(tmp_path)
        shutil.copy(url, root)
        LandCoverAI(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            LandCoverAI(str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LandCoverAI(split="foo")

    def test_plot(self, dataset: LandCoverAI) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
