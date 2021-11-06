# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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
from torchgeo.datasets import SeasonalContrastS2Dataset


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestSeasonalContrastS2Dataset:
    @pytest.fixture(
        params=zip(["100k", "1m"], [["B1"], SeasonalContrastS2Dataset.ALL_BANDS])
    )
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> SeasonalContrastS2Dataset:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            SeasonalContrastS2Dataset,
            "md5s",
            {
                "100k": "4d3e6e4afed7e581b7de1bfa2f7c29da",
                "1m": "3bb3fcf90f5de7d5781ce0cb85fd20af",
            },
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            SeasonalContrastS2Dataset,
            "urls",
            {
                "100k": os.path.join("tests", "data", "seco", "seco_100k.zip"),
                "1m": os.path.join("tests", "data", "seco", "seco_1m.zip"),
            },
        )
        root = str(tmp_path)
        version, bands = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return SeasonalContrastS2Dataset(
            root, version, bands, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: SeasonalContrastS2Dataset) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)

    def test_len(self, dataset: SeasonalContrastS2Dataset) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: SeasonalContrastS2Dataset) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: SeasonalContrastS2Dataset) -> None:
        SeasonalContrastS2Dataset(root=dataset.root, download=True)

    def test_invalid_version(self) -> None:
        with pytest.raises(AssertionError):
            SeasonalContrastS2Dataset(version="foo")

    def test_invalid_band(self) -> None:
        with pytest.raises(AssertionError):
            SeasonalContrastS2Dataset(bands=["A1steaksauce"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            SeasonalContrastS2Dataset(str(tmp_path))
