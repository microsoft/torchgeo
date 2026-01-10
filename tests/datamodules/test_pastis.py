# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest

from torchgeo.datamodules import PASTISDataModule, PASTISR100DataModule


class TestPASTISDataModule:
    @pytest.fixture
    def datamodule(self) -> PASTISDataModule:
        root = os.path.join('tests', 'data', 'pastis')
        return PASTISDataModule(root=root, batch_size=2, num_workers=0, mode='semantic')

    def test_train_dataloader(self, datamodule: PASTISDataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: PASTISDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: PASTISDataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))


class TestPASTISR100DataModule:
    @pytest.fixture
    def datamodule(self, tmp_path: Path) -> PASTISR100DataModule:
        src = os.path.join('tests', 'data', 'pastis', 'PASTIS-R')
        dst = tmp_path / 'PASTIS-R-100'
        shutil.copytree(src, dst)
        return PASTISR100DataModule(
            root=tmp_path, batch_size=2, num_workers=0, mode='semantic'
        )

    def test_train_dataloader(self, datamodule: PASTISR100DataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: PASTISR100DataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: PASTISR100DataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))

