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

from torchgeo.datasets import XView2


class TestXView2:
    @pytest.fixture(params=["train", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> XView2:
        monkeypatch.setattr(
            XView2,
            "metadata",
            {
                "train": {
                    "filename": "train_images_labels_targets.tar.gz",
                    "md5": "373e61d55c1b294aa76b94dbbd81332b",
                    "directory": "train",
                },
                "test": {
                    "filename": "test_images_labels_targets.tar.gz",
                    "md5": "bc6de81c956a3bada38b5b4e246266a1",
                    "directory": "test",
                },
            },
        )
        root = os.path.join("tests", "data", "xview2")
        split = request.param
        transforms = nn.Identity()
        return XView2(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: XView2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: XView2) -> None:
        assert len(dataset) == 2

    def test_extract(self, tmp_path: Path) -> None:
        shutil.copyfile(
            os.path.join(
                "tests", "data", "xview2", "train_images_labels_targets.tar.gz"
            ),
            os.path.join(tmp_path, "train_images_labels_targets.tar.gz"),
        )
        shutil.copyfile(
            os.path.join(
                "tests", "data", "xview2", "test_images_labels_targets.tar.gz"
            ),
            os.path.join(tmp_path, "test_images_labels_targets.tar.gz"),
        )
        XView2(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(
            os.path.join(tmp_path, "train_images_labels_targets.tar.gz"), "w"
        ) as f:
            f.write("bad")
        with open(
            os.path.join(tmp_path, "test_images_labels_targets.tar.gz"), "w"
        ) as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            XView2(root=str(tmp_path), checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            XView2(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in `root` directory"):
            XView2(str(tmp_path))

    def test_plot(self, dataset: XView2) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"][0].clone()
        dataset.plot(x)
        plt.close()
