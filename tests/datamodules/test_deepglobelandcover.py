# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import DeepGlobeLandCoverDataModule


class TestDeepGlobeLandCoverDataModule:
    @pytest.fixture(scope="class", params=[0.0, 0.5])
    def datamodule(self, request: SubRequest) -> DeepGlobeLandCoverDataModule:
        root = os.path.join("tests", "data", "deepglobelandcover")
        batch_size = 1
        num_workers = 0
        val_split_size = request.param
        dm = DeepGlobeLandCoverDataModule(
            root, batch_size, num_workers, val_split_pct=val_split_size
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: DeepGlobeLandCoverDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: DeepGlobeLandCoverDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: DeepGlobeLandCoverDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
