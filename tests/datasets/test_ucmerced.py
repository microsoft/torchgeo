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
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import UCMerced


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestUCMerced:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> UCMerced:
        monkeypatch.setattr(torchgeo.datasets.ucmerced, "download_url", download_url)
        md5 = "a42ef8779469d196d8f2971ee135f030"
        monkeypatch.setattr(UCMerced, "md5", md5)
        url = os.path.join("tests", "data", "ucmerced", "UCMerced_LandUse.zip")
        monkeypatch.setattr(UCMerced, "url", url)
        monkeypatch.setattr(
            UCMerced,
            "split_urls",
            {
                "train": os.path.join(
                    "tests", "data", "ucmerced", "uc_merced-train.txt"
                ),
                "val": os.path.join("tests", "data", "ucmerced", "uc_merced-val.txt"),
                "test": os.path.join("tests", "data", "ucmerced", "uc_merced-test.txt"),
            },
        )
        monkeypatch.setattr(
            UCMerced,
            "split_md5s",
            {
                "train": "a01fa9f13333bb176fc1bfe26ff4c711",
                "val": "a01fa9f13333bb176fc1bfe26ff4c711",
                "test": "a01fa9f13333bb176fc1bfe26ff4c711",
            },
        )
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return UCMerced(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: UCMerced) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: UCMerced) -> None:
        assert len(dataset) == 4

    def test_add(self, dataset: UCMerced) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 8

    def test_already_downloaded(self, dataset: UCMerced, tmp_path: Path) -> None:
        UCMerced(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: UCMerced, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        UCMerced(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            UCMerced(str(tmp_path))

    def test_plot(self, dataset: UCMerced) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
