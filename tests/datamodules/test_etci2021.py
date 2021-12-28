# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import ETCI2021DataModule


class TestETCI2021DataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> ETCI2021DataModule:
        root = os.path.join("tests", "data", "etci2021")
        seed = 0
        batch_size = 2
        num_workers = 0
        dm = ETCI2021DataModule(root, seed, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: ETCI2021DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: ETCI2021DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: ETCI2021DataModule) -> None:
        next(iter(datamodule.test_dataloader()))
