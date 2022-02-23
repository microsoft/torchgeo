# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import USAVars

pytest.importorskip("pandas", minversion="0.19.1")


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestUSAVars:
    @pytest.fixture()
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> USAVars:

        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.usavars, "download_url", download_url
        )

        md5 = "b504580a00bdc27097d5421dec50481b"
        monkeypatch.setattr(USAVars, "md5", md5)  # type: ignore[attr-defined]

        data_url = os.path.join("tests", "data", "usavars", "usavars.zip")
        monkeypatch.setattr(USAVars, "data_url", data_url)  # type: ignore[attr-defined]

        label_urls = {
            "elevation": os.path.join("tests", "data", "usavars", "elevation.csv"),
            "population": os.path.join("tests", "data", "usavars", "population.csv"),
            "treecover": os.path.join("tests", "data", "usavars", "treecover.csv"),
        }
        monkeypatch.setattr(  # type: ignore[attr-defined]
            USAVars, "label_urls", label_urls
        )

        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]

        return USAVars(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: USAVars) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].ndim == 3
        assert len(x.keys()) == 4  # image, elevation, population, treecover
        assert x["image"].shape[0] == 4  # R, G, B, Inf

    def test_len(self, dataset: USAVars) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: USAVars) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: USAVars) -> None:
        USAVars(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "usavars", "usavars.zip")
        root = str(tmp_path)
        shutil.copy(pathname, root)
        for csv in ["elevation.csv", "population.csv", "treecover.csv"]:
            shutil.copy(os.path.join("tests", "data", "usavars", csv), root)

        USAVars(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            USAVars(str(tmp_path))

    def test_plot(self, dataset: USAVars) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()
