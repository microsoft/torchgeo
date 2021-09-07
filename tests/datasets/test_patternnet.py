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
from torchgeo.datasets import PatternNet


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestPatternNet:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> PatternNet:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "5649754c78219a2c19074ff93666cc61"
        monkeypatch.setattr(PatternNet, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "patternnet", "PatternNet.zip")
        monkeypatch.setattr(PatternNet, "url", url)  # type: ignore[attr-defined]
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return PatternNet(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: PatternNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3

    def test_len(self, dataset: PatternNet) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: PatternNet) -> None:
        PatternNet(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            PatternNet(str(tmp_path))
