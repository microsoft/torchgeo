# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import Vaihingen2D


class TestVaihingen2D:
    @pytest.fixture(params=["train", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> Vaihingen2D:
        md5s = ["c15fbff78d307e51c73f609c0859afc3", "ec2c0a5149f2371479b38cf8cfbab961"]
        splits = {
            "train": ["top_mosaic_09cm_area1.tif", "top_mosaic_09cm_area11.tif"],
            "test": ["top_mosaic_09cm_area6.tif", "top_mosaic_09cm_area24.tif"],
        }
        monkeypatch.setattr(Vaihingen2D, "md5s", md5s)
        monkeypatch.setattr(Vaihingen2D, "splits", splits)
        root = os.path.join("tests", "data", "vaihingen")
        split = request.param
        transforms = nn.Identity()
        return Vaihingen2D(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: Vaihingen2D) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].ndim == 3
        assert x["mask"].ndim == 2

    def test_len(self, dataset: Vaihingen2D) -> None:
        assert len(dataset) == 2

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join("tests", "data", "vaihingen")
        filenames = [
            "ISPRS_semantic_labeling_Vaihingen.zip",
            "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip",
        ]
        for filename in filenames:
            shutil.copyfile(
                os.path.join(root, filename), os.path.join(str(tmp_path), filename)
            )
        Vaihingen2D(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        filenames = [
            "ISPRS_semantic_labeling_Vaihingen.zip",
            "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip",
        ]
        for filename in filenames:
            with open(os.path.join(tmp_path, filename), "w") as f:
                f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            Vaihingen2D(root=str(tmp_path), checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            Vaihingen2D(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in `root` directory"):
            Vaihingen2D(str(tmp_path))

    def test_plot(self, dataset: Vaihingen2D) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
