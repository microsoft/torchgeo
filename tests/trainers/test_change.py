# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, OSCDDataModule
from torchgeo.datasets import RGBBandsMissingError
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ChangeDetectionTask


class ChangeDetectionTestModel(Module):
    def __init__(self, in_channels: int = 3, classes: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


def create_model(**kwargs: Any) -> Module:
    return ChangeDetectionTestModel(**kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    return None


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class TestChangeDetectionTask:
    @pytest.mark.parametrize('name', ['oscd'])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(smp, 'Unet', create_model)
        monkeypatch.setattr(
            OSCDDataModule, 'predict_dataloader', OSCDDataModule.test_dataloader
        )

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        # multiply in_chans by 2 since images are concatenated
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans'] * 2
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    @pytest.mark.parametrize('model', [6], indirect=True)
    def test_weight_file(self, checkpoint: str) -> None:
        ChangeDetectionTask(backbone='resnet18', weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=mocked_weights.meta['model'],
            weights=mocked_weights,
            in_channels=mocked_weights.meta['in_chans'],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=mocked_weights.meta['model'],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.parametrize('model_name', ['unet', 'fcsiamdiff', 'fcsiamconc'])
    @pytest.mark.parametrize(
        'backbone', ['resnet18', 'mobilenet_v2', 'efficientnet-b0']
    )
    def test_freeze_backbone(
        self, model_name: Literal['unet', 'fcsiamdiff', 'fcsiamconc'], backbone: str
    ) -> None:
        model = ChangeDetectionTask(
            model=model_name, backbone=backbone, freeze_backbone=True
        )
        assert all(
            [param.requires_grad is False for param in model.model.encoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.decoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )

    @pytest.mark.parametrize('model_name', ['unet', 'fcsiamdiff', 'fcsiamconc'])
    def test_freeze_decoder(
        self, model_name: Literal['unet', 'fcsiamdiff', 'fcsiamconc']
    ) -> None:
        model = ChangeDetectionTask(model=model_name, freeze_decoder=True)
        assert all(
            [param.requires_grad is False for param in model.model.decoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.encoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )

    @pytest.mark.parametrize('loss_fn', ['bce', 'jaccard', 'focal'])
    def test_losses(self, loss_fn: Literal['bce', 'jaccard', 'focal']) -> None:
        ChangeDetectionTask(loss=loss_fn)

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(OSCDDataModule, 'plot', plot)
        datamodule = OSCDDataModule(
            root='tests/data/oscd',
            batch_size=2,
            patch_size=32,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(backbone='resnet18', in_channels=13, model='unet')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(OSCDDataModule, 'plot', plot_missing_bands)
        datamodule = OSCDDataModule(
            root='tests/data/oscd',
            batch_size=2,
            patch_size=32,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(backbone='resnet18', in_channels=13, model='unet')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)
