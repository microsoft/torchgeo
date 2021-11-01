# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.trainers import CycloneDataModule


@pytest.fixture(scope="module")
def datamodule() -> CycloneDataModule:
    root = os.path.join("tests", "data", "cyclone")
    seed = 0
    batch_size = 1
    num_workers = 0
    dm = CycloneDataModule(root, seed, batch_size, num_workers)
    dm.prepare_data()
    dm.setup()
    return dm


class TestCycloneDataModule:
    def test_train_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
