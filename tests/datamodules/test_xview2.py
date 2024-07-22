# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import XView2DataModule


class TestXView2DataModule:
    @pytest.fixture
    def datamodule(self) -> XView2DataModule:
        root = os.path.join('tests', 'data', 'xview2')
        batch_size = 1
        num_workers = 0
        dm = XView2DataModule(
            root=root, batch_size=batch_size, num_workers=num_workers, val_split_pct=0.5
        )
        dm.prepare_data()
        return dm

    def test_train_dataloader(self, datamodule: XView2DataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: XView2DataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: XView2DataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))
