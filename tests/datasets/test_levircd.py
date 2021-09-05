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
from torchgeo.datasets import LEVIRCDPlus
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLEVIRCDPlus:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> LEVIRCDPlus:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "b61c300e9fd7146eb2c8e2512c0e9d39"
        monkeypatch.setattr(LEVIRCDPlus, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "levircd", "LEVIR-CD+.zip")
        monkeypatch.setattr(LEVIRCDPlus, "url", url)  # type: ignore[attr-defined]
        root = str(tmp_path)
        split = request.param
        transforms = Identity()
        return LEVIRCDPlus(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LEVIRCDPlus) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 2

    def test_len(self, dataset: LEVIRCDPlus) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LEVIRCDPlus) -> None:
        LEVIRCDPlus(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LEVIRCDPlus(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            LEVIRCDPlus(str(tmp_path))
