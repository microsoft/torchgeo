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

from torchgeo.datasets import MillionAID


class TestMillionAID:
    @pytest.fixture(
        scope="class", params=zip(["train", "test"], ["multi-class", "multi-label"])
    )
    def dataset(self, request: SubRequest) -> MillionAID:
        root = os.path.join("tests", "data", "millionaid")
        split, task = request.param
        transforms = nn.Identity()
        return MillionAID(
            root=root, split=split, task=task, transforms=transforms, checksum=True
        )

    def test_getitem(self, dataset: MillionAID) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["image"].ndim == 3

    def test_len(self, dataset: MillionAID) -> None:
        assert len(dataset) == 2

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            MillionAID(str(tmp_path))

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join("tests", "data", "millionaid", "train.zip")
        shutil.copy(url, tmp_path)
        MillionAID(str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "train.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            MillionAID(str(tmp_path), checksum=True)

    def test_plot(self, dataset: MillionAID) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: MillionAID) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
