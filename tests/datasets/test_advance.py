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
from torchgeo.datasets import ADVANCE


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestADVANCE:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ADVANCE:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "advance")
        urls = [
            os.path.join(data_dir, "ADVANCE_vision.zip"),
            os.path.join(data_dir, "ADVANCE_sound.zip"),
        ]
        md5s = ["43acacecebecd17a82bc2c1e719fd7e4", "039b7baa47879a8a4e32b9dd8287f6ad"]
        monkeypatch.setattr(ADVANCE, "urls", urls)
        monkeypatch.setattr(ADVANCE, "md5s", md5s)
        root = str(tmp_path)
        transforms = nn.Identity()
        return ADVANCE(root, transforms, download=True, checksum=True)

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "scipy.io":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_getitem(self, dataset: ADVANCE) -> None:
        pytest.importorskip("scipy", minversion="1.2")
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["audio"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["image"].ndim == 3
        assert x["audio"].shape[0] == 1
        assert x["audio"].ndim == 2
        assert x["label"].ndim == 0

    def test_len(self, dataset: ADVANCE) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: ADVANCE) -> None:
        ADVANCE(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            ADVANCE(str(tmp_path))

    def test_mock_missing_module(
        self, dataset: ADVANCE, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="scipy is not installed and is required to use this dataset",
        ):
            dataset[0]

    def test_plot(self, dataset: ADVANCE) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
