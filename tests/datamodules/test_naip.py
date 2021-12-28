# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import NAIPChesapeakeDataModule


class TestNAIPChesapeakeDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> NAIPChesapeakeDataModule:
        dm = NAIPChesapeakeDataModule(
            os.path.join("tests", "data", "naip"),
            os.path.join("tests", "data", "chesapeake", "BAYWIDE"),
            batch_size=2,
            num_workers=0,
        )
        dm.patch_size = 32
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: NAIPChesapeakeDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: NAIPChesapeakeDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: NAIPChesapeakeDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
