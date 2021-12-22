# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import LoveDADataModule


class TestLoveDADataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> LoveDADataModule:
        root = os.path.join("tests", "data", "loveda")
        batch_size = 2
        num_workers = 0
        scene = ["rural", "urban"]

        dm = LoveDADataModule(
            root_dir=root, scene=scene, batch_size=batch_size, num_workers=num_workers
        )

        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: LoveDADataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: LoveDADataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: LoveDADataModule) -> None:
        next(iter(datamodule.test_dataloader()))
