# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from torchgeo.datamodules import ChesapeakeCVPRDataModule
from torchgeo.datasets import ChesapeakeCVPRTileDataset


class TestChesapeakeCVPRDataModule:
    @pytest.fixture
    def datamodule(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> ChesapeakeCVPRDataModule:
        monkeypatch.setattr(
            ChesapeakeCVPRTileDataset,
            '_files',
            ['de_1m_2013_extended-debuffered-test_tiles', 'spatial_index.geojson'],
        )

        shutil.copytree(
            os.path.join('tests', 'data', 'chesapeake', 'cvpr'),
            tmp_path,
            dirs_exist_ok=True,
        )

        return ChesapeakeCVPRDataModule(
            root=tmp_path,
            train_splits=['de-train'],
            val_splits=['de-val'],
            test_splits=['de-test'],
            batch_size=2,
            patch_size=16,
            length=4,
            num_workers=0,
        )

    def test_train_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        datamodule.setup('fit')
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))
        assert 'image' in batch
        assert 'mask' in batch

    def test_val_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        datamodule.setup('validate')
        dataloader = datamodule.val_dataloader()
        batch = next(iter(dataloader))
        assert 'image' in batch
        assert 'mask' in batch

    def test_test_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        datamodule.setup('test')
        dataloader = datamodule.test_dataloader()
        batch = next(iter(dataloader))
        assert 'image' in batch
        assert 'mask' in batch

    def test_invalid_param_config(self) -> None:
        with pytest.raises(ValueError, match='The pre-generated prior labels'):
            ChesapeakeCVPRDataModule(
                root=os.path.join('tests', 'data', 'chesapeake', 'cvpr'),
                train_splits=['de-test'],
                val_splits=['de-test'],
                test_splits=['de-test'],
                batch_size=2,
                patch_size=32,
                length=4,
                num_workers=0,
                class_set=7,
                use_prior_labels=True,
            )
