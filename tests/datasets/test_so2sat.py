# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import So2Sat

pytest.importorskip("h5py", minversion="2.6")


class TestSo2Sat:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> So2Sat:
        md5s = {
            "train": "82e0f2d51766b89cb905dbaf8275eb5b",
            "validation": "bf292ae4737c1698b1a3c6f5e742e0e1",
            "test": "9a3bbe181b038d4e51f122c4be3c569e",
        }

        monkeypatch.setattr(So2Sat, "md5s", md5s)
        root = os.path.join("tests", "data", "so2sat")
        split = request.param
        transforms = nn.Identity()
        return So2Sat(root=root, split=split, transforms=transforms, checksum=True)

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "h5py":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_getitem(self, dataset: So2Sat) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: So2Sat) -> None:
        assert len(dataset) == 1

    def test_out_of_bounds(self, dataset: So2Sat) -> None:
        # h5py at version 2.10.0 raises a ValueError instead of an IndexError so we
        # check for both here
        with pytest.raises((IndexError, ValueError)):
            dataset[1]

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            So2Sat(split="foo")

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            So2Sat(bands=("OK", "BK"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            So2Sat(str(tmp_path))

    def test_plot(self, dataset: So2Sat) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_rgb(self, dataset: So2Sat) -> None:
        dataset = So2Sat(root=dataset.root, bands=("B03",))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle="Single Band")

    def test_mock_missing_module(
        self, dataset: So2Sat, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="h5py is not installed and is required to use this dataset",
        ):
            So2Sat(dataset.root)
