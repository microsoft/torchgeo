# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import CDL, BoundingBox, ZipDataset
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestCDL:
    @pytest.fixture
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> CDL:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5s = [
            (2021, "0693f0bb10deb79c69bcafe4aa1635b7"),
            (2020, "7695292902a8672d16ac034d4d560d84"),
        ]
        monkeypatch.setattr(CDL, "md5s", md5s)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "cdl", "{}_30m_cdls.zip")
        monkeypatch.setattr(CDL, "url", url)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        root = str(tmp_path)
        transforms = Identity()
        return CDL(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: CDL) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_add(self, dataset: CDL) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_already_downloaded(self, dataset: CDL) -> None:
        CDL(root=dataset.root, download=True)

    def test_plot(self, dataset: CDL) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["mask"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CDL(str(tmp_path))

    def test_invalid_query(self, dataset: CDL) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
