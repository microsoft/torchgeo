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
from matplotlib import pyplot as plt
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
        md5 = "d6ebaae1ea0f3ae960af31531d394521"
        monkeypatch.setattr(OSCD, "md5", md5)  # type: ignore[attr-defined]
        urls = {
            "Onera Satellite Change Detection dataset - Images.zip": os.path.join(
                "tests",
                "data",
                "oscd",
                "Onera Satellite Change Detection dataset - Images.zip",
            ),
            "Onera Satellite Change Detection dataset - Train Labels.zip": os.path.join(
                "tests",
                "data",
                "oscd",
                "Onera Satellite Change Detection dataset - Train Labels.zip",
            ),
            "Onera Satellite Change Detection dataset - Test Labels.zip": os.path.join(
                "tests",
                "data",
                "oscd",
                "Onera Satellite Change Detection dataset - Test Labels.zip",
            ),
        }
        monkeypatch.setattr(OSCD, "urls", urls)  # type: ignore[attr-defined]

        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return OSCD(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: OSCD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].ndim == 4
        assert x["image"].shape[:2] == (2, 13)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["mask"].ndim == 2

    def test_len(self, dataset: OSCD) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: OSCD) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: OSCD) -> None:
        OSCD(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "oscd", "*Onera*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        OSCD(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            OSCD(str(tmp_path))

    def test_plot(self, dataset: OSCD) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()
