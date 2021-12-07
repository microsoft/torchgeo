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
import torchgeo.datasets.utils
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torchgeo.datasets import LoveDA


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLoveDA:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> LoveDA:
        monkeypatch.setattr(
            torchgeo.datasets.utils, "download_url", download_url
        )  # type: ignore[attr-defined]
        md5 = "3d5b1373ef9a3084ec493b9b2056fe07"

        info_dict = {
            "train": {
                "url": os.path.join("tests", "data", "loveda", "Train.zip"),
                "filename": "Train.zip",
                "md5": md5,
            },
            "val": {
                "url": os.path.join("tests", "data", "loveda", "Val.zip"),
                "filename": "Val.zip",
                "md5": md5,
            },
            "test": {
                "url": os.path.join("tests", "data", "loveda", "Test.zip"),
                "filename": "Test.zip",
                "md5": md5,
            },
        }

        monkeypatch.setattr(
            LoveDA, "info_dict", info_dict
        )  # type: ignore[attr-defined]

        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return LoveDA(
            root=root, split=split, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: LoveDA) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].shape[0] == 3

        if dataset.split != "test":
            assert isinstance(x["mask"], torch.Tensor)
            assert x["image"].shape[-2:] == x["mask"].shape[-2:]
        else:
            assert "mask" not in x

    def test_len(self, dataset: LoveDA) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LoveDA) -> None:
        print(dataset.root)
        LoveDA(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LoveDA(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(
            RuntimeError, match="Dataset not found at root directory or corrupted."
        ):
            LoveDA(str(tmp_path))

    def test_plot(self, dataset: LoveDA) -> None:
        # dataset does not have a batch size dimension
        img = dataset[0]["image"].unsqueeze(0)
        if dataset.split != "test":  # training and validation images
            mask = dataset[0]["mask"].unsqueeze(0)
            batch = {"image": img, "mask": mask}
        else:  # test images
            batch = {"image": dataset[0]["image"].unsqueeze(0)}
        dataset.plot(batch, suptitle="Test")
        plt.close()

        # now testing with batch size of 2
        if dataset.split != "test":
            batch = {
                "image": torch.cat((img, img), dim=0),  # type: ignore[attr-defined]
                "mask": torch.cat((mask, mask), dim=0),  # type: ignore[attr-defined]
            }
        else:
            batch = {
                "image": torch.cat((img, img), dim=0)
            }  # type: ignore[attr-defined]
        dataset.plot(batch, suptitle="Test")
        plt.close()
