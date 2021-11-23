# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import OSCD, OSCDDataModule


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestOSCD:
    @pytest.fixture(params=zip(["all", "rgb"], ["train", "test"]))
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> OSCD:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.oscd, "download_url", download_url
        )
        md5 = "d6ebaae1ea0f3ae960af31531d394521"
        monkeypatch.setattr(OSCD, "md5", md5)  # type: ignore[attr-defined]
        urls = {
            "Onera Satellite Change Detection dataset - Images.zip": os.path.join(
                "tests",
                "data",
                "oscd",
                "Onera Satellite Change Detection dataset - Images.zip",
            ),
            "Onera Satellite Change Detection dataset - Train Labels.zip": os.path.join(
                "tests",
                "data",
                "oscd",
                "Onera Satellite Change Detection dataset - Train Labels.zip",
            ),
            "Onera Satellite Change Detection dataset - Test Labels.zip": os.path.join(
                "tests",
                "data",
                "oscd",
                "Onera Satellite Change Detection dataset - Test Labels.zip",
            ),
        }
        monkeypatch.setattr(OSCD, "urls", urls)  # type: ignore[attr-defined]

        bands, split = request.param
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return OSCD(
            root, split, bands, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: OSCD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].ndim == 4
        assert isinstance(x["mask"], torch.Tensor)
        assert x["mask"].ndim == 2

        if dataset.bands == "rgb":
            assert x["image"].shape[:2] == (2, 3)
        else:
            assert x["image"].shape[:2] == (2, 13)

    def test_len(self, dataset: OSCD) -> None:
        if dataset.split == "train":
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_add(self, dataset: OSCD) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: OSCD) -> None:
        OSCD(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "oscd", "*Onera*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        OSCD(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            OSCD(str(tmp_path))

    def test_plot(self, dataset: OSCD) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()


class TestOSCDDataModule:
    @pytest.fixture(scope="class", params=zip(["all", "rgb"], [0.0, 0.5]))
    def datamodule(self, request: SubRequest) -> OSCDDataModule:
        bands, val_split_pct = request.param
        crop_size = (2, 2)
        root = os.path.join("tests", "data", "oscd")
        batch_size = 1
        num_workers = 0
        dm = OSCDDataModule(
            root, bands, batch_size, num_workers, val_split_pct, crop_size
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: OSCDDataModule) -> None:
        sample = next(iter(datamodule.train_dataloader()))
        assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (2, 2)

    def test_val_dataloader(self, datamodule: OSCDDataModule) -> None:
        sample = next(iter(datamodule.val_dataloader()))
        assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (2, 2)

    def test_test_dataloader(self, datamodule: OSCDDataModule) -> None:
        sample = next(iter(datamodule.test_dataloader()))
        assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (3, 3)
