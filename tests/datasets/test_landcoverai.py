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
from torchgeo.datasets import LandCoverAINonGeo


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestLandCoverAINonGeo:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LandCoverAINonGeo:
        pytest.importorskip("cv2", minversion="3.4.2.17")
        monkeypatch.setattr(torchgeo.datasets.landcoverai, "download_url", download_url)
        md5 = "ff8998857cc8511f644d3f7d0f3688d0"
        monkeypatch.setattr(LandCoverAINonGeo, "md5", md5)
        url = os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip")
        monkeypatch.setattr(LandCoverAINonGeo, "url", url)
        sha256 = "ecec8e871faf1bbd8ca525ca95ddc1c1f5213f40afb94599884bd85f990ebd6b"
        monkeypatch.setattr(LandCoverAINonGeo, "sha256", sha256)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return LandCoverAINonGeo(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LandCoverAINonGeo) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: LandCoverAINonGeo) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: LandCoverAINonGeo) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: LandCoverAINonGeo) -> None:
        LandCoverAINonGeo(root=dataset.root, download=True)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        pytest.importorskip("cv2", minversion="3.4.2.17")
        sha256 = "ecec8e871faf1bbd8ca525ca95ddc1c1f5213f40afb94599884bd85f990ebd6b"
        monkeypatch.setattr(LandCoverAINonGeo, "sha256", sha256)
        url = os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip")
        root = str(tmp_path)
        shutil.copy(url, root)
        LandCoverAINonGeo(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            LandCoverAINonGeo(str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LandCoverAINonGeo(split="foo")

    def test_plot(self, dataset: LandCoverAINonGeo) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
