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

import torchgeo.datasets.utils
from torchgeo.datasets import BigEarthNet


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestBigEarthNet:
    @pytest.fixture(
        params=zip(["all", "s1", "s2"], [43, 19, 19], ["train", "val", "test"])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BigEarthNet:
        monkeypatch.setattr(torchgeo.datasets.bigearthnet, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "bigearthnet")
        metadata = {
            "s1": {
                "url": os.path.join(data_dir, "BigEarthNet-S1-v1.0.tar.gz"),
                "md5": "5a64e9ce38deb036a435a7b59494924c",
                "filename": "BigEarthNet-S1-v1.0.tar.gz",
                "directory": "BigEarthNet-S1-v1.0",
            },
            "s2": {
                "url": os.path.join(data_dir, "BigEarthNet-S2-v1.0.tar.gz"),
                "md5": "ef5f41129b8308ca178b04d7538dbacf",
                "filename": "BigEarthNet-S2-v1.0.tar.gz",
                "directory": "BigEarthNet-v1.0",
            },
        }
        splits_metadata = {
            "train": {
                "url": os.path.join(data_dir, "bigearthnet-train.csv"),
                "filename": "bigearthnet-train.csv",
                "md5": "167ac4d5de8dde7b5aeaa812f42031e7",
            },
            "val": {
                "url": os.path.join(data_dir, "bigearthnet-val.csv"),
                "filename": "bigearthnet-val.csv",
                "md5": "aff594ba256a52e839a3b5fefeb9ef42",
            },
            "test": {
                "url": os.path.join(data_dir, "bigearthnet-test.csv"),
                "filename": "bigearthnet-test.csv",
                "md5": "851a6bdda484d47f60e121352dcb1bf5",
            },
        }
        monkeypatch.setattr(BigEarthNet, "metadata", metadata)
        monkeypatch.setattr(BigEarthNet, "splits_metadata", splits_metadata)
        bands, num_classes, split = request.param
        root = str(tmp_path)
        transforms = nn.Identity()
        return BigEarthNet(
            root, split, bands, num_classes, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: BigEarthNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["label"].shape == (dataset.num_classes,)
        assert x["image"].dtype == torch.int32
        assert x["label"].dtype == torch.int64

        if dataset.bands == "all":
            assert x["image"].shape == (14, 120, 120)
        elif dataset.bands == "s1":
            assert x["image"].shape == (2, 120, 120)
        else:
            assert x["image"].shape == (12, 120, 120)

    def test_len(self, dataset: BigEarthNet) -> None:
        if dataset.split == "train":
            assert len(dataset) == 2
        elif dataset.split == "val":
            assert len(dataset) == 1
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: BigEarthNet, tmp_path: Path) -> None:
        BigEarthNet(
            root=str(tmp_path),
            bands=dataset.bands,
            split=dataset.split,
            num_classes=dataset.num_classes,
            download=True,
        )

    def test_already_downloaded_not_extracted(
        self, dataset: BigEarthNet, tmp_path: Path
    ) -> None:
        if dataset.bands == "all":
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata["s1"]["directory"])
            )
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata["s2"]["directory"])
            )
            download_url(dataset.metadata["s1"]["url"], root=str(tmp_path))
            download_url(dataset.metadata["s2"]["url"], root=str(tmp_path))
        elif dataset.bands == "s1":
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata["s1"]["directory"])
            )
            download_url(dataset.metadata["s1"]["url"], root=str(tmp_path))
        else:
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata["s2"]["directory"])
            )
            download_url(dataset.metadata["s2"]["url"], root=str(tmp_path))

        BigEarthNet(
            root=str(tmp_path),
            bands=dataset.bands,
            split=dataset.split,
            num_classes=dataset.num_classes,
            download=False,
        )

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            BigEarthNet(str(tmp_path))

    def test_plot(self, dataset: BigEarthNet) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()
