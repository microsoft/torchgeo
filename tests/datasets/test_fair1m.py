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
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import FAIR1M, DatasetNotFoundError


def download_url(url: str, root: str, filename: str, *args: str, **kwargs: str) -> None:
    os.makedirs(root, exist_ok=True)
    shutil.copy(url, os.path.join(root, filename))


class TestFAIR1M:
    test_root = os.path.join("tests", "data", "fair1m")

    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FAIR1M:
        monkeypatch.setattr(torchgeo.datasets.fair1m, "download_url", download_url)
        urls = {
            "train": (
                os.path.join(self.test_root, "train", "part1", "images.zip"),
                os.path.join(self.test_root, "train", "part1", "labelXml.zip"),
                os.path.join(self.test_root, "train", "part2", "images.zip"),
                os.path.join(self.test_root, "train", "part2", "labelXmls.zip"),
            ),
            "val": (
                os.path.join(self.test_root, "validation", "images.zip"),
                os.path.join(self.test_root, "validation", "labelXmls.zip"),
            ),
            "test": (
                os.path.join(self.test_root, "test", "images0.zip"),
                os.path.join(self.test_root, "test", "images1.zip"),
                os.path.join(self.test_root, "test", "images2.zip"),
            ),
        }
        md5s = {
            "train": (
                "ffbe9329e51ae83161ce24b5b46dc934",
                "2db6fbe64be6ebb0a03656da6c6effe7",
                "401b0f1d75d9d23f2e088bfeaf274cfa",
                "d62b18eae8c3201f6112c2e9db84d605",
            ),
            "val": (
                "83d2f06574fc7158ded0eb1fb256c8fe",
                "316490b200503c54cf43835a341b6dbe",
            ),
            "test": (
                "3c02845752667b96a5749c90c7fdc994",
                "9359107f1d0abac6a5b98725f4064bc0",
                "d7bc2985c625ffd47d86cdabb2a9d2bc",
            ),
        }
        monkeypatch.setattr(FAIR1M, "urls", urls)
        monkeypatch.setattr(FAIR1M, "md5s", md5s)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return FAIR1M(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: FAIR1M) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].shape[0] == 3

        if dataset.split != "test":
            assert isinstance(x["boxes"], torch.Tensor)
            assert isinstance(x["label"], torch.Tensor)
            assert x["boxes"].shape[-2:] == (5, 2)
            assert x["label"].ndim == 1

    def test_len(self, dataset: FAIR1M) -> None:
        if dataset.split == "train":
            assert len(dataset) == 8
        else:
            assert len(dataset) == 4

    def test_already_downloaded(self, dataset: FAIR1M, tmp_path: Path) -> None:
        FAIR1M(root=str(tmp_path), split=dataset.split, download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: FAIR1M, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        for filepath, url in zip(
            dataset.paths[dataset.split], dataset.urls[dataset.split]
        ):
            output = os.path.join(str(tmp_path), filepath)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            download_url(url, root=os.path.dirname(output), filename=output)

        FAIR1M(root=str(tmp_path), split=dataset.split, checksum=True)

    def test_corrupted(self, tmp_path: Path, dataset: FAIR1M) -> None:
        md5s = tuple(["randomhash"] * len(FAIR1M.md5s[dataset.split]))
        FAIR1M.md5s[dataset.split] = md5s
        shutil.rmtree(dataset.root)
        for filepath, url in zip(
            dataset.paths[dataset.split], dataset.urls[dataset.split]
        ):
            output = os.path.join(str(tmp_path), filepath)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            download_url(url, root=os.path.dirname(output), filename=output)

        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            FAIR1M(root=str(tmp_path), split=dataset.split, checksum=True)

    def test_not_downloaded(self, tmp_path: Path, dataset: FAIR1M) -> None:
        shutil.rmtree(str(tmp_path))
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            FAIR1M(root=str(tmp_path), split=dataset.split)

    def test_plot(self, dataset: FAIR1M) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        if dataset.split != "test":
            x["prediction_boxes"] = x["boxes"].clone()
            dataset.plot(x)
            plt.close()
