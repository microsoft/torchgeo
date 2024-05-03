# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
import torchvision.transforms.functional as F
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch import Tensor
from torchvision.transforms import InterpolationMode

import torchgeo.datasets.utils
from torchgeo.datamodules import LEVIRCDDataModule, LEVIRCDPlusDataModule
from torchgeo.datasets import LEVIRCD, LEVIRCDPlus


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


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
    def datamodule(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> LEVIRCDPlusDataModule:
        monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', download_url)
        md5 = '0ccca34310bfe7096dadfbf05b0d180f'
        monkeypatch.setattr(LEVIRCDPlus, 'md5', md5)
        url = os.path.join('tests', 'data', 'levircd', 'levircdplus', 'LEVIR-CD+.zip')
        monkeypatch.setattr(LEVIRCDPlus, 'url', url)

        root = str(tmp_path)
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
    def datamodule(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> LEVIRCDDataModule:
        directory = os.path.join('tests', 'data', 'levircd', 'levircd')
        splits = {
            'train': {
                'url': os.path.join(directory, 'train.zip'),
                'filename': 'train.zip',
                'md5': '7c2e24b3072095519f1be7eb01fae4ff',
            },
            'val': {
                'url': os.path.join(directory, 'val.zip'),
                'filename': 'val.zip',
                'md5': '5c320223ba88b6fc8ff9d1feebc3b84e',
            },
            'test': {
                'url': os.path.join(directory, 'test.zip'),
                'filename': 'test.zip',
                'md5': '021db72d4486726d6a0702563a617b32',
            },
        }
        monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', download_url)
        monkeypatch.setattr(LEVIRCD, 'splits', splits)

        root = str(tmp_path)
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
