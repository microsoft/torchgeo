# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Tuple, cast

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.trainers import BigEarthNetDataModule


class TestBigEarthNetDataModule:
    @pytest.fixture(
        params=[
            ("s1", True),
            ("s2", True),
            ("all", True),
            ("s1", False),
            ("s2", False),
            ("all", False),
        ]
    )
    def datamodule(self, request: SubRequest) -> BigEarthNetDataModule:
        bands, unsupervised_mode = request.param
        root = os.path.join("tests", "data", "bigearthnet")
        batch_size = 1
        num_workers = 0
        dm = BigEarthNetDataModule(
            root,
            bands,
            batch_size,
            num_workers,
            unsupervised_mode,
            val_split_pct=0.3,
            test_split_pct=0.3,
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
