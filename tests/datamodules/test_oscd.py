# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest
from pytorch_lightning import Trainer

from torchgeo.datamodules import OSCDDataModule


class TestOSCDDataModule:
    @pytest.fixture(scope="class", params=["all", "rgb"])
    def datamodule(self, request: SubRequest) -> OSCDDataModule:
        bands = request.param
        num_tiles_per_batch = 1
        num_patches_per_tile = 2
        patch_size = 2
        root = os.path.join("tests", "data", "oscd")
        num_workers = 0
        dm = OSCDDataModule(
            root=root,
            download=True,
            bands=bands,
            num_tiles_per_batch=num_tiles_per_batch,
            num_patches_per_tile=num_patches_per_tile,
            patch_size=patch_size,
            val_split_pct=0.5,
            num_workers=num_workers,
        )
        dm.prepare_data()
        dm.trainer = Trainer(max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: OSCDDataModule) -> None:
        datamodule.setup("fit")
        datamodule.trainer.training = True  # type: ignore[union-attr]
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
        assert batch["image"].shape[0] == batch["mask"].shape[0] == 2
        if datamodule.bands == "all":
            assert batch["image"].shape[1] == 26
        else:
            assert batch["image"].shape[1] == 6

    def test_val_dataloader(self, datamodule: OSCDDataModule) -> None:
        datamodule.setup("validate")
        datamodule.trainer.validating = True  # type: ignore[union-attr]
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        if datamodule.val_split_pct > 0.0:
            assert batch["image"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
            assert batch["image"].shape[0] == batch["mask"].shape[0] == 1024
            if datamodule.bands == "all":
                assert batch["image"].shape[1] == 26
            else:
                assert batch["image"].shape[1] == 6

    def test_test_dataloader(self, datamodule: OSCDDataModule) -> None:
        datamodule.setup("test")
        datamodule.trainer.testing = True  # type: ignore[union-attr]
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch["image"].shape[-2:] == batch["mask"].shape[-2:] == (2, 2)
        assert batch["image"].shape[0] == batch["mask"].shape[0] == 1024
        if datamodule.bands == "all":
            assert batch["image"].shape[1] == 26
        else:
            assert batch["image"].shape[1] == 6
