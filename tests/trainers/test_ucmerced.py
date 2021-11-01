# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.trainers import UCMercedDataModule


@pytest.fixture(scope="module")
def datamodule() -> UCMercedDataModule:
    root = os.path.join("tests", "data", "ucmerced")
    batch_size = 2
    num_workers = 0
    dm = UCMercedDataModule(root, batch_size, num_workers)
    dm.prepare_data()
    dm.setup()
    return dm


class TestUCMercedDataModule:
    def test_train_dataloader(self, datamodule: UCMercedDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: UCMercedDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: UCMercedDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
