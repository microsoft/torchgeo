# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import NASAMarineDebrisDataModule


class TestNASAMarineDebrisDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> NASAMarineDebrisDataModule:
        root = os.path.join("tests", "data", "nasa_marine_debris")
        batch_size = 2
        num_workers = 0
        val_split_pct = 0.3
        test_split_pct = 0.3
        dm = NASAMarineDebrisDataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=val_split_pct,
            test_split_pct=test_split_pct,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: NASAMarineDebrisDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: NASAMarineDebrisDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: NASAMarineDebrisDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
