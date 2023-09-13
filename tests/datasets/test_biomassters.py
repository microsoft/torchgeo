# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import shutil
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import BioMassters


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestBioMassters:
    @pytest.fixture(
        params=product(["train", "test"], [["S1"], ["S2"], ["S1", "S2"], [True, False]])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BioMassters:
        split, sensor, as_time_series = request.param
        monkeypatch.setattr(torchgeo.datasets.biomassters, "download_url", download_url)

        md5s = {}
        monkeypatch.setattr(BioMassters, "md5s", md5s)
        url = os.path.join("tests", "data", "biomassters")
        monkeypatch.setattr(BioMassters, "url", url)

        root = str(tmp_path)

        return BioMassters(
            root,
            split=split,
            sensor=sensor,
            as_time_series=as_time_series,
            download=True,
            checksum=True,
        )

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            BioMassters(split="foo")

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            BioMassters(sensor=["S3"])

    def test_already_downloaded(self, dataset: BioMassters, tmp_path: Path) -> None:
        BioMassters(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: BioMassters, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        BioMassters(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            BioMassters(str(tmp_path))
