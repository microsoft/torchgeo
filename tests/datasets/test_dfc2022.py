# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import DFC2022


class TestDFC2022:
    @pytest.fixture(params=["train", "train-unlabeled", "val"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], request: SubRequest
    ) -> DFC2022:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            DFC2022,
            "metadata",
            {
                "train": {
                    "filename": "labeled_train.zip",
                    "md5": "a794b2bb9cf3907d2e209966de0d6501",
                    "directory": "labeled_train",
                },
                "train-unlabeled": {
                    "filename": "unlabeled_train.zip",
                    "md5": "ca2417ce3c030842026afa095b0f0b1d",
                    "directory": "unlabeled_train",
                },
                "val": {
                    "filename": "val.zip",
                    "md5": "5ae50e2c0d3da12cf5ea3473bb4c3c3e",
                    "directory": "val",
                },
            },
        )
        root = os.path.join("tests", "data", "dfc2022")
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return DFC2022(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: DFC2022) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].ndim == 3
        assert x["image"].shape[0] == 4

        if dataset.split == "train":
            assert isinstance(x["mask"], torch.Tensor)
            assert x["mask"].ndim == 2

    def test_len(self, dataset: DFC2022) -> None:
        assert len(dataset) == 2

    def test_extract(self, tmp_path: Path) -> None:
        shutil.copyfile(
            os.path.join("tests", "data", "dfc2022", "labeled_train.zip"),
            os.path.join(tmp_path, "labeled_train.zip"),
        )
        shutil.copyfile(
            os.path.join("tests", "data", "dfc2022", "unlabeled_train.zip"),
            os.path.join(tmp_path, "unlabeled_train.zip"),
        )
        shutil.copyfile(
            os.path.join("tests", "data", "dfc2022", "val.zip"),
            os.path.join(tmp_path, "val.zip"),
        )
        DFC2022(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "labeled_train.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            DFC2022(root=str(tmp_path), checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            DFC2022(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in `root` directory"):
            DFC2022(str(tmp_path))

    def test_plot(self, dataset: DFC2022) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        if dataset.split == "train":
            x["prediction"] = x["mask"].clone()
            dataset.plot(x)
            plt.close()
            del x["mask"]
            dataset.plot(x)
            plt.close()
