# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import SeasonalContrastS2


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSeasonalContrastS2:
    @pytest.fixture(params=zip(["100k", "1m"], [["B1"], SeasonalContrastS2.ALL_BANDS]))
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> SeasonalContrastS2:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.seco, "download_url", download_url
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            SeasonalContrastS2,
            "md5s",
            {
                "100k": "4d3e6e4afed7e581b7de1bfa2f7c29da",
                "1m": "3bb3fcf90f5de7d5781ce0cb85fd20af",
            },
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            SeasonalContrastS2,
            "urls",
            {
                "100k": os.path.join("tests", "data", "seco", "seco_100k.zip"),
                "1m": os.path.join("tests", "data", "seco", "seco_1m.zip"),
            },
        )
        root = str(tmp_path)
        version, bands = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return SeasonalContrastS2(
            root, version, bands, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: SeasonalContrastS2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)

    def test_len(self, dataset: SeasonalContrastS2) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: SeasonalContrastS2) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: SeasonalContrastS2) -> None:
        SeasonalContrastS2(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "seco", "*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        SeasonalContrastS2(root)

    def test_invalid_version(self) -> None:
        with pytest.raises(AssertionError):
            SeasonalContrastS2(version="foo")

    def test_invalid_band(self) -> None:
        with pytest.raises(AssertionError):
            SeasonalContrastS2(bands=["A1steaksauce"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SeasonalContrastS2(str(tmp_path))
