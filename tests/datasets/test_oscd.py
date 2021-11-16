# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import OSCD


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestOSCD:
    @pytest.fixture
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> OSCD:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.oscd, "download_url", download_url
        )
        # idk if this is right yet
        md5 = "1adf156f628aa32fb2e8fe6cada16c04"

        monkeypatch.setattr(OSCD, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "oscd", "OSCD.zip")
        monkeypatch.setattr(OSCD, "url", url)  # type: ignore[attr-defined]

        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return OSCD(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: OSCD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: OSCD) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: OSCD) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: OSCD) -> None:
        OSCD(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "oscd", "OSCD.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        OSCD(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            OSCD(str(tmp_path))
