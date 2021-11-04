# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import BigEarthNet, BigEarthNetDataModule


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestBigEarthNet:
    @pytest.fixture(
        params=product(["all", "s1", "s2"], [43, 19], ["train", "val", "test"])
    )
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> BigEarthNet:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.bigearthnet, "download_url", download_url
        )
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
        monkeypatch.setattr(  # type: ignore[attr-defined]
            BigEarthNet, "metadata", metadata
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            BigEarthNet, "splits_metadata", splits_metadata
        )
        bands, num_classes, split = request.param
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return BigEarthNet(
            root, split, bands, num_classes, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: BigEarthNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["label"].shape == (dataset.num_classes,)
        assert x["image"].dtype == torch.int32  # type: ignore[attr-defined]
        assert x["label"].dtype == torch.int64  # type: ignore[attr-defined]

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
        "to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            BigEarthNet(str(tmp_path))


class TestBigEarthNetDataModule:
    @pytest.fixture(scope="class", params=["s1", "s2", "all"])
    def datamodule(self, request: SubRequest) -> BigEarthNetDataModule:
        bands = request.param
        root = os.path.join("tests", "data", "bigearthnet")
        num_classes = 19
        batch_size = 1
        num_workers = 0
        dm = BigEarthNetDataModule(
            root,
            bands,
            num_classes,
            batch_size,
            num_workers,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: BigEarthNetDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: BigEarthNetDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: BigEarthNetDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
