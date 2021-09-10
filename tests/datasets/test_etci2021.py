# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import ETCI2021
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestETCI2021:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> ETCI2021:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5s = [
            "50c10eb07d6db9aee3ba36401e4a2c45",
            "3e8b5a3cb95e6029e0e2c2d4b4ec6fba",
            "c8ee1e5d3e478761cd00ebc6f28b0ae7",
        ]
        data_dir = os.path.join("tests", "data", "etci2021")
        urls = [
            os.path.join(data_dir, "train.zip"),
            os.path.join(data_dir, "val_with_ref_labels.zip"),
            os.path.join(data_dir, "test_without_ref_labels.zip"),
        ]
        monkeypatch.setattr(ETCI2021, "md5s", md5s)  # type: ignore[attr-defined]
        monkeypatch.setattr(ETCI2021, "urls", urls)  # type: ignore[attr-defined]
        root = str(tmp_path)
        split = request.param
        transforms = Identity()
        return ETCI2021(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: ETCI2021) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 6
        assert x["image"].shape[-2:] == x["mask"].shape[-2:]

        if dataset.split != "test":
            assert x["mask"].ndim == 3
        else:
            assert x["mask"].ndim == 2

    def test_len(self, dataset: ETCI2021) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: ETCI2021) -> None:
        ETCI2021(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            ETCI2021(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            ETCI2021(str(tmp_path))
