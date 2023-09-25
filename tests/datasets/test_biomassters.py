# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import shutil
from itertools import product
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import BioMassters


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestBioMassters:
    @pytest.fixture(
        params=product(["train", "test"], [["S1"], ["S2"], ["S1", "S2"]], [True, False])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BioMassters:
        split, sensors, as_time_series = request.param
        monkeypatch.setattr(torchgeo.datasets.biomassters, "download_url", download_url)

        md5s = {
            "train_features": "a1d9e9d620e1341448707b5a4baef68c",
            "test_features": "e7cd859b1e031f94e645d01cd07c1966",
            "train_agbm": "b6fbbb594c9ba683b25a77f5ff5ace97",
        }
        monkeypatch.setattr(BioMassters, "md5s", md5s)
        url = os.path.join("tests", "data", "biomassters", "{}")
        monkeypatch.setattr(BioMassters, "url", url)

        root = str(tmp_path)

        return BioMassters(
            root,
            split=split,
            sensors=sensors,
            as_time_series=as_time_series,
            download=True,
            checksum=True,
        )

    def test_invalid_split(self, dataset: BioMassters) -> None:
        with pytest.raises(AssertionError):
            BioMassters(dataset.root, split="foo")

    def test_invalid_bands(self, dataset: BioMassters) -> None:
        with pytest.raises(AssertionError):
            BioMassters(dataset.root, sensors=["S3"])

    def test_already_downloaded(self, dataset: BioMassters, tmp_path: Path) -> None:
        BioMassters(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: BioMassters, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        shutil.copytree(os.path.join("tests", "data", "biomassters"), str(tmp_path))
        BioMassters(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        match = "Dataset not found"
        with pytest.raises(RuntimeError, match=match):
            BioMassters(str(tmp_path))
