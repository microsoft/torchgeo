# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest
from lightning.pytorch import Trainer
from pytest import MonkeyPatch

from torchgeo.datamodules import LEVIRCDPlusDataModule


class TestLEVIRCDPlusDataModule:
    @pytest.fixture
    def datamodule(
        self, monkeypatch: MonkeyPatch, request: SubRequest
    ) -> LEVIRCDPlusDataModule:
        monkeypatch.setattr(LEVIRCDPlusDataModule, "download", Mock(return_value=True))

        root = os.path.join("tests", "data", "LEVIR-CD+")
        dm = LEVIRCDPlusDataModule(root=root, download=True, num_workers=0)
        dm.prepare_data()
        dm.trainer = Trainer(accelerator="cpu", max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup("fit")
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image1"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
        assert batch["image1"].shape[0] == batch["mask"].shape[0] == 1
        assert batch["image2"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
        assert batch["image2"].shape[0] == batch["mask"].shape[0] == 1
        assert batch["image1"].shape[1] == 3
        assert batch["image2"].shape[1] == 3

    def test_val_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup("validate")
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        if datamodule.val_split_pct > 0.0:
            assert batch["image1"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
            assert batch["image1"].shape[0] == batch["mask"].shape[0] == 1
            assert batch["image2"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
            assert batch["image2"].shape[0] == batch["mask"].shape[0] == 1
            assert batch["image1"].shape[1] == 3
            assert batch["image2"].shape[1] == 3

    def test_test_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup("test")
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image1"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
        assert batch["image1"].shape[0] == batch["mask"].shape[0] == 1
        assert batch["image2"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
        assert batch["image2"].shape[0] == batch["mask"].shape[0] == 1
        assert batch["image1"].shape[1] == 3
        assert batch["image2"].shape[1] == 3
