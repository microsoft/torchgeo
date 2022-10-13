# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import VHR10DataModule


class TestVHR10DataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> VHR10DataModule:
        root = os.path.join("tests", "data", "vhr10")
        batch_size = 1
        num_workers = 0
        val_split_pct = 0.4
        test_split_pct = 0.2
        dm = VHR10DataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=val_split_pct,
            test_split_pct=test_split_pct,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: VHR10DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: VHR10DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: VHR10DataModule) -> None:
        next(iter(datamodule.test_dataloader()))
