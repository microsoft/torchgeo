# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Test suite for PatternNetDataModule."""

import os

import pytest

from torchgeo.datamodules import PatternNetDataModule


class TestPatternNetDataModule:
    @pytest.fixture
    def datamodule(self) -> PatternNetDataModule:
        root = os.path.join('tests', 'data', 'patternnet')
        dm = PatternNetDataModule(
            root=root,
            batch_size=1,
            num_workers=0,
            val_split_pct=0.2,
            test_split_pct=0.1,
            download=False,
        )
        return dm

    def test_train_dataloader(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup('fit')
        batch = next(iter(datamodule.train_dataloader()))
        assert isinstance(batch, dict)
        assert 'image' in batch and 'label' in batch

    def test_val_dataloader(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        assert isinstance(batch, dict)
        assert 'image' in batch and 'label' in batch

    def test_test_dataloader(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup('test')
        batch = next(iter(datamodule.test_dataloader()))
        assert isinstance(batch, dict)
        assert 'image' in batch and 'label' in batch
