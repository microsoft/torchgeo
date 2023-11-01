# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
from lightning.pytorch import Trainer
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datamodules import LEVIRCDPlusDataModule
from torchgeo.datasets import LEVIRCDPlus


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLEVIRCDPlusDataModule:
    @pytest.fixture(params=["train", "validate", "test"])
    def datamodule(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> LEVIRCDPlusDataModule:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        md5 = "1adf156f628aa32fb2e8fe6cada16c04"
        monkeypatch.setattr(LEVIRCDPlus, "md5", md5)
        url = os.path.join("tests", "data", "levircd", "LEVIR-CD+.zip")
        monkeypatch.setattr(LEVIRCDPlus, "url", url)

        root = str(tmp_path)
        dm = LEVIRCDPlusDataModule(
            root=root, download=True, num_workers=0, checksum=True
        )
        dm.prepare_data()
        dm.trainer = Trainer(accelerator="cpu", max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup("fit")
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image1"].shape[-2:] == batch["mask"].shape[-2:] == (256, 256)
        assert batch["image1"].shape[0] == batch["mask"].shape[0] == 4
        assert batch["image2"].shape[-2:] == batch["mask"].shape[-2:] == (256, 256)
        assert batch["image2"].shape[0] == batch["mask"].shape[0] == 4
        assert batch["image1"].shape[1] == 3
        assert batch["image2"].shape[1] == 3

    def test_val_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup("validate")
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        if datamodule.val_split_pct > 0.0:
            assert batch["image1"].shape[-2:] == batch["mask"].shape[-2:] == (256, 256)
            assert batch["image1"].shape[0] == batch["mask"].shape[0] == 4
            assert batch["image2"].shape[-2:] == batch["mask"].shape[-2:] == (256, 256)
            assert batch["image2"].shape[0] == batch["mask"].shape[0] == 4
            assert batch["image1"].shape[1] == 3
            assert batch["image2"].shape[1] == 3

    def test_test_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup("test")
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image1"].shape[-2:] == batch["mask"].shape[-2:] == (256, 256)
        assert batch["image1"].shape[0] == batch["mask"].shape[0] == 1
        assert batch["image2"].shape[-2:] == batch["mask"].shape[-2:] == (256, 256)
        assert batch["image2"].shape[0] == batch["mask"].shape[0] == 1
        assert batch["image1"].shape[1] == 3
        assert batch["image2"].shape[1] == 3
