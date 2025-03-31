# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest
from lightning.pytorch import Trainer

from torchgeo.datamodules import OSCDDataModule
from torchgeo.datasets import OSCD


class TestOSCDDataModule:
    @pytest.fixture(params=[OSCD.all_bands, OSCD.rgb_bands])
    def datamodule(self, request: SubRequest) -> OSCDDataModule:
        bands = request.param
        root = os.path.join('tests', 'data', 'oscd')
        dm = OSCDDataModule(
            root=root,
            download=True,
            bands=bands,
            batch_size=2,
            patch_size=8,
            val_split_pct=0.5,
            num_workers=0,
        )
        dm.prepare_data()
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: OSCDDataModule) -> None:
        datamodule.setup('fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (8, 8)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (8, 8)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 1
        if datamodule.bands == OSCD.all_bands:
            assert batch['image1'].shape[1] == 13
            assert batch['image2'].shape[1] == 13
        else:
            assert batch['image1'].shape[1] == 3
            assert batch['image2'].shape[1] == 3

    def test_val_dataloader(self, datamodule: OSCDDataModule) -> None:
        datamodule.setup('validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        if datamodule.val_split_pct > 0.0:
            assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (8, 8)
            assert batch['image1'].shape[0] == batch['mask'].shape[0] == 64
            assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (8, 8)
            assert batch['image2'].shape[0] == batch['mask'].shape[0] == 64
            if datamodule.bands == OSCD.all_bands:
                assert batch['image1'].shape[1] == 13
                assert batch['image2'].shape[1] == 13
            else:
                assert batch['image1'].shape[1] == 3
                assert batch['image2'].shape[1] == 3

    def test_test_dataloader(self, datamodule: OSCDDataModule) -> None:
        datamodule.setup('test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (8, 8)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 64
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (8, 8)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 64
        if datamodule.bands == OSCD.all_bands:
            assert batch['image1'].shape[1] == 13
            assert batch['image2'].shape[1] == 13
        else:
            assert batch['image1'].shape[1] == 3
            assert batch['image2'].shape[1] == 3
