# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import BigEarthNet


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestBigEarthNet:
    @pytest.fixture()
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> BigEarthNet:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.bigearthnet, "download_url", download_url
        )
        md5 = "ef5f41129b8308ca178b04d7538dbacf"
        monkeypatch.setattr(BigEarthNet, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "bigearthnet", "BigEarthNet-S2-v1.0.tar.gz")
        monkeypatch.setattr(BigEarthNet, "url", url)  # type: ignore[attr-defined]
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return BigEarthNet(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: BigEarthNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape == (12, 120, 120)
        assert x["image"].dtype == torch.int32  # type: ignore[attr-defined]
        assert x["label"].shape == (43,)
        assert x["label"].dtype == torch.int64  # type: ignore[attr-defined]

    def test_len(self, dataset: BigEarthNet) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: BigEarthNet, tmp_path: Path) -> None:
        BigEarthNet(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: BigEarthNet, tmp_path: Path
    ) -> None:
        shutil.rmtree(os.path.join(dataset.root, dataset.directory))
        download_url(dataset.url, root=str(tmp_path))
        BigEarthNet(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            BigEarthNet(str(tmp_path))
