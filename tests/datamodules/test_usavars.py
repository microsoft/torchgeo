# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import USAVarsDataModule


class TestUSAVarsDataModule:
    @pytest.fixture(
        scope="class",
        params=zip(
            [["elevation", "population"], ["treecover"]],
            [True, False],
            [(0.5, 0.0), (0.0, 0.5)],
        ),
    )
    def datamodule(self, request: SubRequest) -> USAVarsDataModule:
        labels, fixed_shuffle, split = request.param
        val_split_pct, test_split_pct = split
        root = os.path.join("tests", "data", "usavars")
        batch_size = 1
        num_workers = 0

        dm = USAVarsDataModule(
            root,
            labels,
            None,
            fixed_shuffle,
            batch_size,
            num_workers,
            val_split_pct,
            test_split_pct,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: USAVarsDataModule) -> None:
        assert len(datamodule.train_dataloader()) == 1
        sample = next(iter(datamodule.train_dataloader()))
        assert sample["labels"].shape[1] == len(datamodule.labels)
        if datamodule.fixed_shuffle:
            assert sample["labels"][0, 0] == 1.0

    def test_val_dataloader(self, datamodule: USAVarsDataModule) -> None:
        if datamodule.val_split_pct == 0.5:
            assert len(datamodule.val_dataloader()) == 1
            sample = next(iter(datamodule.val_dataloader()))
            assert sample["labels"].shape[1] == len(datamodule.labels)
            if datamodule.fixed_shuffle:
                assert sample["labels"][0, 0] == 0.0
        else:
            assert len(datamodule.val_dataloader()) == 0

    def test_test_dataloader(self, datamodule: USAVarsDataModule) -> None:
        if datamodule.test_split_pct == 0.5:
            assert len(datamodule.test_dataloader()) == 1
            sample = next(iter(datamodule.test_dataloader()))
            assert sample["labels"].shape[1] == len(datamodule.labels)
            if datamodule.fixed_shuffle:
                assert sample["labels"][0, 0] == 0.0
        else:
            assert len(datamodule.test_dataloader()) == 0
