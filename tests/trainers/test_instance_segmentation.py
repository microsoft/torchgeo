# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import timm
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, SEN12MSDataModule
from torchgeo.datasets import RGBBandsMissingError
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import InstanceSegmentationTask


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
            'ftw',
            'geonrw',
            'gid15',
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
            'spacenet6',
            'ssl4eo_l_benchmark_cdl',
            'ssl4eo_l_benchmark_nlcd',
            'vaihingen2d',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

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
        InstanceSegmentationTask(backbone='resnet18', weights=checkpoint, num_classes=6)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        InstanceSegmentationTask(
            backbone=mocked_weights.meta['model'],
            weights=mocked_weights,
            in_channels=mocked_weights.meta['in_chans'],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        InstanceSegmentationTask(
            backbone=mocked_weights.meta['model'],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        InstanceSegmentationTask(
            backbone=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        InstanceSegmentationTask(
            backbone=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    def test_invalid_model(self) -> None:
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            InstanceSegmentationTask(model='invalid_model')

    def test_invalid_loss(self) -> None:
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            InstanceSegmentationTask(loss='invalid_loss')

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(SEN12MSDataModule, 'plot', plot)
        datamodule = SEN12MSDataModule(
            root='tests/data/sen12ms', batch_size=1, num_workers=0
        )
        model = InstanceSegmentationTask(
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
        model = InstanceSegmentationTask(
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
        model = InstanceSegmentationTask(
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
