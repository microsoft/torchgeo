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


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
class TestRESISC45:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> RESISC45:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.resisc45, "download_url", download_url
        )
        md5 = "5d898bd91e3ebc64314893ff191b2f9d"
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
        assert len(dataset) == 9

    def test_already_downloaded(self, dataset: RESISC45, tmp_path: Path) -> None:
        RESISC45(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: RESISC45, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        RESISC45(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            RESISC45(str(tmp_path))
