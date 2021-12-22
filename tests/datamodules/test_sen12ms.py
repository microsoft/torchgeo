# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import SEN12MSDataModule


class TestSEN12MSDataModule:
    @pytest.fixture(scope="class", params=["all", "s1", "s2-all", "s2-reduced"])
    def datamodule(self, request: SubRequest) -> SEN12MSDataModule:
        root = os.path.join("tests", "data", "sen12ms")
        seed = 0
        bands = request.param
        batch_size = 1
        num_workers = 0
        dm = SEN12MSDataModule(root, seed, bands, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: SEN12MSDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: SEN12MSDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: SEN12MSDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
