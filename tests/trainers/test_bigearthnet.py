# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.trainers import BigEarthNetDataModule


class TestBigEarthNetDataModule:
    @pytest.fixture(scope="class", params=["s1", "s2", "all"])
    def datamodule(self, request: SubRequest) -> BigEarthNetDataModule:
        bands = request.param
        root = os.path.join("tests", "data", "bigearthnet")
        num_classes = 19
        batch_size = 1
        num_workers = 0
        dm = BigEarthNetDataModule(
            root,
            bands,
            num_classes,
            batch_size,
            num_workers,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: BigEarthNetDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: BigEarthNetDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: BigEarthNetDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
