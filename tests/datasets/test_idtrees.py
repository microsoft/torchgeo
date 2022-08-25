# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import IDTReeS

pytest.importorskip("pandas", minversion="0.23.2")
pytest.importorskip("laspy", minversion="2")


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestIDTReeS:
    @pytest.fixture(params=zip(["train", "test", "test"], ["task1", "task1", "task2"]))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> IDTReeS:
        monkeypatch.setattr(torchgeo.datasets.idtrees, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "idtrees")
        metadata = {
            "train": {
                "url": os.path.join(data_dir, "IDTREES_competition_train_v2.zip"),
                "md5": "5ddfa76240b4bb6b4a7861d1d31c299c",
                "filename": "IDTREES_competition_train_v2.zip",
            },
            "test": {
                "url": os.path.join(data_dir, "IDTREES_competition_test_v2.zip"),
                "md5": "b108931c84a70f2a38a8234290131c9b",
                "filename": "IDTREES_competition_test_v2.zip",
            },
        }
        split, task = request.param
        monkeypatch.setattr(IDTReeS, "metadata", metadata)
        root = str(tmp_path)
        transforms = nn.Identity()
        return IDTReeS(root, split, task, transforms, download=True, checksum=True)

    @pytest.fixture(params=["pandas", "laspy", "open3d"])
    def mock_missing_module(self, monkeypatch: MonkeyPatch, request: SubRequest) -> str:
        import_orig = builtins.__import__
        package = str(request.param)

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        return package

    def test_getitem(self, dataset: IDTReeS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["chm"], torch.Tensor)
        assert isinstance(x["hsi"], torch.Tensor)
        assert isinstance(x["las"], torch.Tensor)
        assert x["image"].shape == (3, 200, 200)
        assert x["chm"].shape == (1, 200, 200)
        assert x["hsi"].shape == (369, 200, 200)
        assert x["las"].ndim == 2
        assert x["las"].shape[0] == 3

        if "label" in x:
            assert isinstance(x["label"], torch.Tensor)
        if "boxes" in x:
            assert isinstance(x["boxes"], torch.Tensor)
            if x["boxes"].ndim != 1:
                assert x["boxes"].ndim == 2
                assert x["boxes"].shape[-1] == 4

    def test_len(self, dataset: IDTReeS) -> None:
        assert len(dataset) == 3

    def test_already_downloaded(self, dataset: IDTReeS) -> None:
        IDTReeS(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            IDTReeS(str(tmp_path))

    def test_not_extracted(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "idtrees", "*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        IDTReeS(root)

    def test_mock_missing_module(
        self, dataset: IDTReeS, mock_missing_module: str
    ) -> None:
        package = mock_missing_module

        if package in ["pandas", "laspy"]:
            with pytest.raises(
                ImportError,
                match=f"{package} is not installed and is required to use this dataset",
            ):
                IDTReeS(dataset.root, dataset.split, dataset.task)
        elif package in ["open3d"]:
            with pytest.raises(
                ImportError,
                match=f"{package} is not installed and is required to plot point cloud",
            ):
                dataset.plot_las(0)

    def test_plot(self, dataset: IDTReeS) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        if "boxes" in x:
            x["prediction_boxes"] = x["boxes"]
            dataset.plot(x, show_titles=True)
            plt.close()
        if "label" in x:
            x["prediction_label"] = x["label"]
            dataset.plot(x, show_titles=False)
            plt.close()

    @pytest.mark.skipif(
        sys.platform in ["darwin", "win32"],
        reason="segmentation fault on macOS and windows",
    )
    def test_plot_las(self, dataset: IDTReeS) -> None:
        pytest.importorskip("open3d", minversion="0.11.2")
        vis = dataset.plot_las(index=0, colormap="BrBG")
        vis.close()
        vis = dataset.plot_las(index=0, colormap=None)
        vis.close()
        vis = dataset.plot_las(index=1, colormap=None)
        vis.close()
