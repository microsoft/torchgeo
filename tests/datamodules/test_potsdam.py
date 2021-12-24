# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import Potsdam2DDataModule


class TestPotsdam2DDataModule:
    @pytest.fixture(scope="class", params=[0.0, 0.5])
    def datamodule(self, request: SubRequest) -> Potsdam2DDataModule:
        root = os.path.join("tests", "data", "potsdam")
        batch_size = 1
        num_workers = 0
        val_split_size = request.param
        dm = Potsdam2DDataModule(
            root, batch_size, num_workers, val_split_pct=val_split_size
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: Potsdam2DDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: Potsdam2DDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: Potsdam2DDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
