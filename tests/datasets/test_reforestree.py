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
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import ReforesTree


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestReforesTree:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ReforesTree:
        pytest.importorskip("pandas", minversion="0.23.2")
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "reforestree")

        url = os.path.join(data_dir, "reforesTree.zip")

        md5 = "387e04dbbb0aa803f72bd6d774409648"

        monkeypatch.setattr(ReforesTree, "url", url)
        monkeypatch.setattr(ReforesTree, "md5", md5)
        root = str(tmp_path)
        transforms = nn.Identity()
        return ReforesTree(
            root=root, transforms=transforms, download=True, checksum=True
        )

    def test_already_downloaded(self, dataset: ReforesTree) -> None:
        ReforesTree(root=dataset.root, download=True)

    def test_getitem(self, dataset: ReforesTree) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert isinstance(x["boxes"], torch.Tensor)
        assert isinstance(x["agb"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["image"].ndim == 3
        assert len(x["boxes"]) == 2

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__
        package = "pandas"

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_mock_missing_module(
        self, dataset: ReforesTree, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="pandas is not installed and is required to use this dataset",
        ):
            ReforesTree(root=dataset.root)

    def test_len(self, dataset: ReforesTree) -> None:
        assert len(dataset) == 2

    def test_not_extracted(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas", minversion="0.23.2")
        url = os.path.join("tests", "data", "reforestree", "reforesTree.zip")
        shutil.copy(url, tmp_path)
        ReforesTree(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "reforesTree.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            ReforesTree(root=str(tmp_path), checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            ReforesTree(str(tmp_path))

    def test_plot(self, dataset: ReforesTree) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: ReforesTree) -> None:
        x = dataset[0].copy()
        x["prediction_boxes"] = x["boxes"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()
