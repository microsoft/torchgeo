# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import SKIPPD


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSKIPPD:
    @pytest.fixture(params=["trainval", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SKIPPD:
        monkeypatch.setattr(torchgeo.datasets.skippd, "download_url", download_url)

        md5 = "1133b2de453a9674776abd7519af5051"
        monkeypatch.setattr(SKIPPD, "md5", md5)
        url = os.path.join("tests", "data", "skippd", "dj417rh1007.zip")
        monkeypatch.setattr(SKIPPD, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return SKIPPD(root, split, transforms, download=True, checksum=True)

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "h5py":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_mock_missing_module(
        self, dataset: SKIPPD, tmp_path: Path, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="h5py is not installed and is required to use this dataset",
        ):
            SKIPPD(dataset.root, download=True, checksum=True)

    def test_already_extracted(self, dataset: SKIPPD) -> None:
        SKIPPD(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "skippd", "dj417rh1007.zip")
        root = str(tmp_path)
        shutil.copy(pathname, root)
        SKIPPD(root)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_getitem(self, dataset: SKIPPD, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert isinstance(x["date"], str)
        assert x["image"].shape == (3, 64, 64)

    def test_len(self, dataset: SKIPPD) -> None:
        assert len(dataset) == 3

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SKIPPD(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            SKIPPD(str(tmp_path))

    def test_plot(self, dataset: SKIPPD) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["label"]
        dataset.plot(sample)
        plt.close()
