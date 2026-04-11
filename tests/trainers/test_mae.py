# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
from pytest import MonkeyPatch
from timm.models import VisionTransformer
from torch.nn import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import SSL4EOS12
from torchgeo.main import main
from torchgeo.models import ViTSmall16_Weights
from torchgeo.trainers import MAETask


def create_model(*args: Any, **kwargs: Any) -> Module:
    kwargs.pop('pretrained', None)
    return VisionTransformer(depth=1, **kwargs)


class TestMAETask:
    @pytest.mark.parametrize('name', ['ssl4eo_s12_mae_1', 'ssl4eo_s12_mae_2'])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        if name.startswith('ssl4eo_s12'):
            monkeypatch.setattr(SSL4EOS12, '__len__', lambda self: 2)

        monkeypatch.setattr(timm, 'create_model', create_model)

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

    def test_full_scheduler(self, monkeypatch: MonkeyPatch) -> None:
        config = os.path.join('tests', 'conf', 'ssl4eo_s12_mae_1.yaml')
        monkeypatch.setattr(SSL4EOS12, '__len__', lambda self: 2)
        monkeypatch.setattr(timm, 'create_model', create_model)

        args = [
            '--config',
            config,
            '--model.init_args.warmup_epochs',
            '0',
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            'True',
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ViTSmall16_Weights.SENTINEL2_ALL_MAE

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

    def test_wrong_model_type(self) -> None:
        with pytest.raises(ValueError, match='is not a ViT architecture'):
            MAETask(model='resnet18', weights=None)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        match = 'num classes .* != num classes in pretrained model'
        with pytest.warns(UserWarning, match=match):
            MAETask(
                model=mocked_weights.meta['model'],
                weights=mocked_weights,
                in_channels=mocked_weights.meta['in_chans'],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        match = 'num classes .* != num classes in pretrained model'
        with pytest.warns(UserWarning, match=match):
            MAETask(
                model=mocked_weights.meta['model'],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta['in_chans'],
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        MAETask(
            model=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        MAETask(
            model=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )
