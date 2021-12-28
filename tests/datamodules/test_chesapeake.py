# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import ChesapeakeCVPRDataModule


class TestChesapeakeCVPRDataModule:
    @pytest.fixture(scope="class", params=[5, 7])
    def datamodule(self, request: SubRequest) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patch_size=32,
            patches_per_tile=2,
            batch_size=2,
            num_workers=0,
            class_set=request.param,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.test_dataloader()))

    def test_nodata_check(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        nodata_check = datamodule.nodata_check(4)
        sample = {
            "image": torch.ones(1, 2, 2),  # type: ignore[attr-defined]
            "mask": torch.ones(2, 2),  # type: ignore[attr-defined]
        }
        out = nodata_check(sample)
        assert torch.equal(  # type: ignore[attr-defined]
            out["image"], torch.zeros(1, 4, 4)  # type: ignore[attr-defined]
        )
        assert torch.equal(  # type: ignore[attr-defined]
            out["mask"], torch.zeros(4, 4)  # type: ignore[attr-defined]
        )
