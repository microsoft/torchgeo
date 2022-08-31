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
from torchgeo.datasets import ZueriCrop

pytest.importorskip("h5py", minversion="2.6")


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestZueriCrop:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ZueriCrop:
        monkeypatch.setattr(torchgeo.datasets.zuericrop, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "zuericrop")
        urls = [
            os.path.join(data_dir, "ZueriCrop.hdf5"),
            os.path.join(data_dir, "labels.csv"),
        ]
        md5s = ["1635231df67f3d25f4f1e62c98e221a4", "5118398c7a5bbc246f5f6bb35d8d529b"]
        monkeypatch.setattr(ZueriCrop, "urls", urls)
        monkeypatch.setattr(ZueriCrop, "md5s", md5s)
        root = str(tmp_path)
        transforms = nn.Identity()
        return ZueriCrop(root=root, transforms=transforms, download=True, checksum=True)

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "h5py":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_getitem(self, dataset: ZueriCrop) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["boxes"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

        # Image tests
        assert x["image"].ndim == 4

        # Instance masks tests
        assert x["mask"].ndim == 3
        assert x["mask"].shape[-2:] == x["image"].shape[-2:]

        # Bboxes tests
        assert x["boxes"].ndim == 2
        assert x["boxes"].shape[1] == 4

        # Labels tests
        assert x["label"].ndim == 1

    def test_len(self, dataset: ZueriCrop) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: ZueriCrop) -> None:
        ZueriCrop(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            ZueriCrop(str(tmp_path))

    def test_mock_missing_module(
        self, dataset: ZueriCrop, tmp_path: Path, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="h5py is not installed and is required to use this dataset",
        ):
            ZueriCrop(dataset.root, download=True, checksum=True)

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            ZueriCrop(bands=("OK", "BK"))

    def test_plot(self, dataset: ZueriCrop) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="prediction")
        plt.close()

    def test_plot_rgb(self, dataset: ZueriCrop) -> None:
        dataset = ZueriCrop(root=dataset.root, bands=("B02",))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], time_step=0, suptitle="Single Band")
