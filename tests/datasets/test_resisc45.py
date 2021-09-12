# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import sys
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import RESISC45

pytest.importorskip("rarfile")


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
class TestRESISC45:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> RESISC45:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "9c221122164d17b8118d2b6527ee5e9c"
        monkeypatch.setattr(RESISC45, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "resisc45", "NWPU-RESISC45.rar")
        monkeypatch.setattr(RESISC45, "url", url)  # type: ignore[attr-defined]
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return RESISC45(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: RESISC45) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3

    def test_len(self, dataset: RESISC45) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: RESISC45) -> None:
        RESISC45(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            RESISC45(str(tmp_path))
