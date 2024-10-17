# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, SEN12MSDataModule
from torchgeo.datasets import LandCoverAI, RGBBandsMissingError
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import SemanticSegmentationTask


class SegmentationTestModel(Module):
    def __init__(self, in_channels: int = 3, classes: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


def create_model(**kwargs: Any) -> Module:
    return SegmentationTestModel(**kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    return None


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class TestSemanticSegmentationTask:
    @pytest.mark.parametrize(
        'name',
        [
            'agrifieldnet',
            'cabuar',
            'chabud',
            'chesapeake_cvpr_5',
            'chesapeake_cvpr_7',
            'deepglobelandcover',
            'etci2021',
            'geonrw',
            'gid15',
            'glacier_calving_front',
            'inria',
            'l7irish',
            'l8biome',
            'landcoverai',
            'landcoverai100',
            'loveda',
            'naipchesapeake',
            'potsdam2d',
            'sen12ms_all',
            'sen12ms_s1',
            'sen12ms_s2_all',
            'sen12ms_s2_reduced',
            'sentinel2_cdl',
            'sentinel2_eurocrops',
            'sentinel2_nccm',
            'sentinel2_south_america_soybean',
            'southafricacroptype',
            'spacenet1',
            'ssl4eo_l_benchmark_cdl',
            'ssl4eo_l_benchmark_nlcd',
            'vaihingen2d',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        match name:
            case 'chabud' | 'cabuar':
                pytest.importorskip('h5py', minversion='3.6')
            case 'landcoverai':
                sha256 = (
                    'ecec8e871faf1bbd8ca525ca95ddc1c1f5213f40afb94599884bd85f990ebd6b'
                )
                monkeypatch.setattr(LandCoverAI, 'sha256', sha256)

        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(smp, 'Unet', create_model)
        monkeypatch.setattr(smp, 'DeepLabV3Plus', create_model)

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
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_weight_file(self, checkpoint: str) -> None:
        SemanticSegmentationTask(backbone='resnet18', weights=checkpoint, num_classes=6)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        SemanticSegmentationTask(
            backbone=mocked_weights.meta['model'],
            weights=mocked_weights,
            in_channels=mocked_weights.meta['in_chans'],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        SemanticSegmentationTask(
            backbone=mocked_weights.meta['model'],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        SemanticSegmentationTask(
            backbone=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        SemanticSegmentationTask(
            backbone=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    def test_invalid_model(self) -> None:
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(model='invalid_model')

    def test_invalid_loss(self) -> None:
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(loss='invalid_loss')

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(SEN12MSDataModule, 'plot', plot)
        datamodule = SEN12MSDataModule(
            root='tests/data/sen12ms', batch_size=1, num_workers=0
        )
        model = SemanticSegmentationTask(
            backbone='resnet18', in_channels=15, num_classes=6
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(SEN12MSDataModule, 'plot', plot_missing_bands)
        datamodule = SEN12MSDataModule(
            root='tests/data/sen12ms', batch_size=1, num_workers=0
        )
        model = SemanticSegmentationTask(
            backbone='resnet18', in_channels=15, num_classes=6
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    @pytest.mark.parametrize('model_name', ['unet', 'deeplabv3+'])
    @pytest.mark.parametrize(
        'backbone', ['resnet18', 'mobilenet_v2', 'efficientnet-b0']
    )
    def test_freeze_backbone(self, model_name: str, backbone: str) -> None:
        model = SemanticSegmentationTask(
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

    @pytest.mark.parametrize('model_name', ['unet', 'deeplabv3+'])
    def test_freeze_decoder(self, model_name: str) -> None:
        model = SemanticSegmentationTask(model=model_name, freeze_decoder=True)
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
