# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest
from pytorch_lightning import Trainer

from torchgeo.datamodules import OSCDDataModule


class TestOSCDDataModule:
    @pytest.fixture(params=["all", "rgb"])
    def datamodule(self, request: SubRequest) -> OSCDDataModule:
        bands = request.param
        root = os.path.join("tests", "data", "oscd")
        dm = OSCDDataModule(
            root=root,
            download=True,
            bands=bands,
            batch_size=1,
            patch_size=2,
            val_split_pct=0.5,
            num_workers=0,
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
            assert batch["image"].shape[0] == batch["mask"].shape[0] == 2
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
        assert batch["image"].shape[0] == batch["mask"].shape[0] == 2
        if datamodule.bands == "all":
            assert batch["image"].shape[1] == 26
        else:
            assert batch["image"].shape[1] == 6
