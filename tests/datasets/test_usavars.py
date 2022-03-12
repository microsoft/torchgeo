# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
import shutil
from pathlib import Path
from typing import Any, Generator

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
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        request: SubRequest,
    ) -> USAVars:

        monkeypatch.setattr(torchgeo.datasets.usavars, "download_url", download_url)

        md5 = "b504580a00bdc27097d5421dec50481b"
        monkeypatch.setattr(USAVars, "md5", md5)

        data_url = os.path.join("tests", "data", "usavars", "uar.zip")
        monkeypatch.setattr(USAVars, "data_url", data_url)

        label_urls = {
            "elevation": os.path.join("tests", "data", "usavars", "elevation.csv"),
            "population": os.path.join("tests", "data", "usavars", "population.csv"),
            "treecover": os.path.join("tests", "data", "usavars", "treecover.csv"),
            "income": os.path.join("tests", "data", "usavars", "income.csv"),
            "nightlights": os.path.join("tests", "data", "usavars", "nightlights.csv"),
            "roads": os.path.join("tests", "data", "usavars", "roads.csv"),
            "housing": os.path.join("tests", "data", "usavars", "housing.csv"),
        }
        monkeypatch.setattr(USAVars, "label_urls", label_urls)

        root = str(tmp_path)
        transforms = nn.Identity()

        return USAVars(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: USAVars) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].ndim == 3
        assert len(x.keys()) == 2  # image, elevation, population, treecover
        assert x["image"].shape[0] == 4  # R, G, B, Inf

    def test_len(self, dataset: USAVars) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: USAVars) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: USAVars) -> None:
        USAVars(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "usavars", "uar.zip")
        root = str(tmp_path)
        shutil.copy(pathname, root)
        csvs = [
            "elevation.csv",
            "population.csv",
            "treecover.csv",
            "income.csv",
            "nightlights.csv",
            "roads.csv",
            "housing.csv",
        ]
        for csv in csvs:
            shutil.copy(os.path.join("tests", "data", "usavars", csv), root)

        USAVars(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            USAVars(str(tmp_path))

    @pytest.fixture(params=["pandas"])
    def mock_missing_module(
        self, monkeypatch: MonkeyPatch, request: SubRequest
    ) -> str:
        import_orig = builtins.__import__
        package = str(request.param)

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        return package

    def test_mock_missing_module(
        self, dataset: USAVars, mock_missing_module: str
    ) -> None:
        package = mock_missing_module
        if package == "pandas":
            with pytest.raises(
                ImportError,
                match=f"{package} is not installed and is required to use this dataset",
            ):
                USAVars(dataset.root)

    def test_plot(self, dataset: USAVars) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()
