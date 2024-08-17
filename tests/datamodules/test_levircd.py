# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torchvision.transforms.functional as F
from lightning.pytorch import Trainer
from torch import Tensor
from torchvision.transforms import InterpolationMode

from torchgeo.datamodules import LEVIRCDDataModule, LEVIRCDPlusDataModule


def transforms(sample: dict[str, Tensor]) -> dict[str, Tensor]:
    sample['image1'] = F.resize(
        sample['image1'],
        size=[1024, 1024],
        antialias=True,
        interpolation=InterpolationMode.BILINEAR,
    )
    sample['image2'] = F.resize(
        sample['image2'],
        size=[1024, 1024],
        antialias=True,
        interpolation=InterpolationMode.BILINEAR,
    )
    sample['mask'] = F.resize(
        sample['mask'].unsqueeze(dim=0),
        size=[1024, 1024],
        interpolation=InterpolationMode.NEAREST,
    )
    return sample


class TestLEVIRCDPlusDataModule:
    @pytest.fixture
    def datamodule(self) -> LEVIRCDPlusDataModule:
        root = os.path.join('tests', 'data', 'levircd', 'levircdplus')
        dm = LEVIRCDPlusDataModule(
            root=root,
            download=True,
            num_workers=0,
            checksum=True,
            val_split_pct=0.5,
            transforms=transforms,
        )
        dm.prepare_data()
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup('fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (256, 256)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 8
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (256, 256)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 8
        assert batch['image1'].shape[1] == 3
        assert batch['image2'].shape[1] == 3

    def test_val_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup('validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        if datamodule.val_split_pct > 0.0:
            assert (
                batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
            )
            assert batch['image1'].shape[0] == batch['mask'].shape[0] == 1
            assert (
                batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
            )
            assert batch['image2'].shape[0] == batch['mask'].shape[0] == 1
            assert batch['image1'].shape[1] == 3
            assert batch['image2'].shape[1] == 3

    def test_test_dataloader(self, datamodule: LEVIRCDPlusDataModule) -> None:
        datamodule.setup('test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image1'].shape[1] == 3
        assert batch['image2'].shape[1] == 3


class TestLEVIRCDDataModule:
    @pytest.fixture
    def datamodule(self) -> LEVIRCDDataModule:
        root = os.path.join('tests', 'data', 'levircd', 'levircd')
        dm = LEVIRCDDataModule(
            root=root,
            download=True,
            num_workers=0,
            checksum=True,
            transforms=transforms,
        )
        dm.prepare_data()
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    def test_train_dataloader(self, datamodule: LEVIRCDDataModule) -> None:
        datamodule.setup('fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (256, 256)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 8
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (256, 256)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 8
        assert batch['image1'].shape[1] == 3
        assert batch['image2'].shape[1] == 3

    def test_val_dataloader(self, datamodule: LEVIRCDDataModule) -> None:
        datamodule.setup('validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image1'].shape[1] == 3
        assert batch['image2'].shape[1] == 3

    def test_test_dataloader(self, datamodule: LEVIRCDDataModule) -> None:
        datamodule.setup('test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        assert batch['image1'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
        assert batch['image1'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image2'].shape[-2:] == batch['mask'].shape[-2:] == (1024, 1024)
        assert batch['image2'].shape[0] == batch['mask'].shape[0] == 1
        assert batch['image1'].shape[1] == 3
        assert batch['image2'].shape[1] == 3
