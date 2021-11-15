# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import EuroSAT, EuroSATDataModule


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestEuroSAT:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> EuroSAT:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.eurosat, "download_url", download_url
        )
        md5 = "aa051207b0547daba0ac6af57808d68e"
        monkeypatch.setattr(EuroSAT, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "eurosat", "EuroSATallBands.zip")
        monkeypatch.setattr(EuroSAT, "url", url)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            EuroSAT,
            "split_urls",
            {
                "train": os.path.join("tests", "data", "eurosat", "eurosat-train.txt"),
                "val": os.path.join("tests", "data", "eurosat", "eurosat-val.txt"),
                "test": os.path.join("tests", "data", "eurosat", "eurosat-test.txt"),
            },
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            EuroSAT,
            "split_md5s",
            {
                "train": "4af60a00fdfdf8500572ae5360694b71",
                "val": "4af60a00fdfdf8500572ae5360694b71",
                "test": "4af60a00fdfdf8500572ae5360694b71",
            },
        )
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return EuroSAT(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: EuroSAT) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: EuroSAT) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: EuroSAT) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: EuroSAT, tmp_path: Path) -> None:
        EuroSAT(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: EuroSAT, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        EuroSAT(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            EuroSAT(str(tmp_path))


class TestEuroSATDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> EuroSATDataModule:
        root = os.path.join("tests", "data", "eurosat")
        batch_size = 1
        num_workers = 0
        dm = EuroSATDataModule(root, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: EuroSATDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: EuroSATDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: EuroSATDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
