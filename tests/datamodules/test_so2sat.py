# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import So2SatDataModule

pytest.importorskip("h5py")


class TestSo2SatDataModule:
    @pytest.fixture(scope="class", params=zip([True, False], ["rgb", "s2"]))
    def datamodule(self, request: SubRequest) -> So2SatDataModule:
        unsupervised_mode, bands = request.param
        root = os.path.join("tests", "data", "so2sat")
        batch_size = 2
        num_workers = 0
        dm = So2SatDataModule(root, batch_size, num_workers, bands, unsupervised_mode)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: So2SatDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: So2SatDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: So2SatDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
