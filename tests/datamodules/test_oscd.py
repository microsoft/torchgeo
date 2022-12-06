# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import OSCDDataModule


class TestOSCDDataModule:
    @pytest.fixture(scope="class", params=zip(["all", "rgb"], [0.0, 0.5]))
    def datamodule(self, request: SubRequest) -> OSCDDataModule:
        bands, val_split_pct = request.param
        patch_size = (2, 2)
        num_patches_per_tile = 2
        root = os.path.join("tests", "data", "oscd")
        batch_size = 1
        num_workers = 0
        dm = OSCDDataModule(
            root=root,
            bands=bands,
            train_batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=val_split_pct,
            patch_size=patch_size,
            num_patches_per_tile=num_patches_per_tile,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: OSCDDataModule) -> None:
        sample = next(iter(datamodule.train_dataloader()))
        assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (2, 2)
        assert sample["image"].shape[0] == sample["mask"].shape[0] == 2
        if datamodule.test_dataset.bands == "all":
            assert sample["image"].shape[1] == 26
        else:
            assert sample["image"].shape[1] == 6

    def test_val_dataloader(self, datamodule: OSCDDataModule) -> None:
        sample = next(iter(datamodule.val_dataloader()))
        if datamodule.val_split_pct > 0.0:
            assert (
                sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (1280, 1280)
            )
            assert sample["image"].shape[0] == sample["mask"].shape[0] == 1
            if datamodule.test_dataset.bands == "all":
                assert sample["image"].shape[1] == 26
            else:
                assert sample["image"].shape[1] == 6

    def test_test_dataloader(self, datamodule: OSCDDataModule) -> None:
        sample = next(iter(datamodule.test_dataloader()))
        assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (1280, 1280)
        assert sample["image"].shape[0] == sample["mask"].shape[0] == 1
        if datamodule.test_dataset.bands == "all":
            assert sample["image"].shape[1] == 26
        else:
            assert sample["image"].shape[1] == 6
