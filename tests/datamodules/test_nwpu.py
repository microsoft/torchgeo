# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import pytest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datamodules import VHR10DataModule
from torchgeo.datasets import VHR10


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestVHR10DataModule:
    @pytest.fixture()
    def datamodule(self, monkeypatch: MonkeyPatch) -> VHR10DataModule:
        root = os.path.join("tests", "data", "vhr10")
        batch_size = 1
        num_workers = 0
        val_split_pct = 0.33
        test_split_pct = 0.33
        dm = VHR10DataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=val_split_pct,
            test_split_pct=test_split_pct,
        )
        monkeypatch.setattr(torchgeo.datasets.nwpu, "download_url", download_url)
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        url = os.path.join("tests", "data", "vhr10", "NWPU VHR-10 dataset.rar")
        monkeypatch.setitem(VHR10.image_meta, "url", url)
        md5 = "1de589590bf1a9bb35c1c35f34229ff2"
        monkeypatch.setitem(VHR10.image_meta, "md5", md5)
        url = os.path.join("tests", "data", "vhr10", "annotations.json")
        monkeypatch.setitem(VHR10.target_meta, "url", url)
        md5 = "9f1d91c5229d31a613d5a5a35ee94f95"
        monkeypatch.setitem(VHR10.target_meta, "md5", md5)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: VHR10DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: VHR10DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: VHR10DataModule) -> None:
        next(iter(datamodule.test_dataloader()))
