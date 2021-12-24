# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import EuroSATDataModule


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
