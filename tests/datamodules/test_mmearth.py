# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest
from lightning.pytorch import Trainer
from torch import Tensor

from torchgeo.datamodules import MMEarthDataModule

pytest.importorskip('h5py', minversion='3.6')


class TestMMEarthDataModule:
    @pytest.fixture(params=['MMEarth', 'MMEarth64', 'MMEarth100k'])
    def datamodule(self, request: SubRequest) -> MMEarthDataModule:
        ds_version = request.param
        root = os.path.join('tests', 'data', 'mmearth')
        dm = MMEarthDataModule(root=root, ds_version=ds_version, num_workers=0)
        dm.prepare_data()
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: MMEarthDataModule) -> None:
        datamodule.setup('fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert 'sentinel2' in batch
        assert isinstance(batch['sentinel2'], Tensor)

    def test_val_dataloader(self, datamodule: MMEarthDataModule) -> None:
        datamodule.setup('validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert 'sentinel2' in batch
        assert isinstance(batch['sentinel2'], Tensor)

    def test_test_dataloader(self, datamodule: MMEarthDataModule) -> None:
        datamodule.setup('test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert 'sentinel2' in batch
        assert isinstance(batch['sentinel2'], Tensor)
