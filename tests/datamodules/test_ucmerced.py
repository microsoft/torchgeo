# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from lightning.pytorch import Trainer

from torchgeo.datamodules import UCMercedDataModule


class TestUCMercedDataModule:
    @pytest.fixture
    def datamodule(self) -> UCMercedDataModule:
        root = os.path.join("tests", "data", "ucmerced")
        dm = UCMercedDataModule(root=root, download=True, batch_size=1, num_workers=0)
        dm.prepare_data()
        dm.trainer = Trainer(accelerator="cpu", max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: UCMercedDataModule) -> None:
        datamodule.setup("fit")
        datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image"].shape[-2:] == (256, 256)

    def test_val_dataloader(self, datamodule: UCMercedDataModule) -> None:
        datamodule.setup("validate")
        datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image"].shape[-2:] == (256, 256)

    def test_test_dataloader(self, datamodule: UCMercedDataModule) -> None:
        datamodule.setup("test")
        datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image"].shape[-2:] == (256, 256)
