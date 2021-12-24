# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import FAIR1MDataModule


class TestFAIR1MDataModule:
    @pytest.fixture(scope="class", params=[True, False])
    def datamodule(self) -> FAIR1MDataModule:
        root = os.path.join("tests", "data", "fair1m")
        batch_size = 2
        num_workers = 0
        dm = FAIR1MDataModule(
            root, batch_size, num_workers, val_split_pct=0.33, test_split_pct=0.33
        )
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
