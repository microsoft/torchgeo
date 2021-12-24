# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import CycloneDataModule


class TestCycloneDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> CycloneDataModule:
        root = os.path.join("tests", "data", "cyclone")
        seed = 0
        batch_size = 1
        num_workers = 0
        dm = CycloneDataModule(root, seed, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
