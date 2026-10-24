# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest
import timm
import torch
from pytest import MonkeyPatch
from torch import nn
from torchvision.models import resnet18
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import SSL4EOS12, SeasonalContrastS2
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.tasks import BYOL
from torchgeo.tasks.byol import BYOLModule, SimCLRAugmentation


class TestBYOLModule:
    def test_custom_augment_fn(self) -> None:
        model = resnet18()
        layer = model.conv1
        new_layer = nn.Conv2d(
            in_channels=4,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        ).requires_grad_()
        model.conv1 = new_layer
        augment_fn = SimCLRAugmentation((2, 2))
        with pytest.warns(DeprecationWarning, match='augment_fn'):
            module = BYOLModule(model, augment_fn=augment_fn)
        with pytest.warns(DeprecationWarning, match='augment'):
            assert module.augment is augment_fn
        replacement = SimCLRAugmentation((2, 2))
        with pytest.warns(DeprecationWarning, match='augment'):
            module.augment = replacement
        assert module.augmentations is replacement

    def test_custom_augmentations(self) -> None:
        model = resnet18()
        augmentations = SimCLRAugmentation((32, 32))
        module = BYOLModule(
            model, image_size=(32, 32), in_channels=3, augmentations=augmentations
        )
        assert module.augmentations is augmentations

    def test_conflicting_augmentations(self) -> None:
        model = resnet18()
        augmentations = SimCLRAugmentation((32, 32))
        with pytest.raises(ValueError, match='cannot be combined'):
            BYOLModule(
                model,
                image_size=(32, 32),
                in_channels=3,
                augmentations=augmentations,
                augment_fn=augmentations,
            )

    def test_load_legacy_augmentation_state_dict(self) -> None:
        def create_module() -> BYOLModule:
            return BYOLModule(
                resnet18(),
                image_size=(32, 32),
                in_channels=3,
                augmentations=nn.BatchNorm2d(3),
            )

        module = create_module()
        legacy_state_dict = {
            key.replace('augmentations.', 'augment.'): value
            for key, value in module.state_dict().items()
        }

        create_module().load_state_dict(legacy_state_dict)


class TestBYOL:
    def test_custom_augmentations(self) -> None:
        augmentations = nn.Identity()
        task = BYOL(model='resnet18', augmentations=augmentations)
        assert task.augmentations is augmentations
        replacement = nn.Identity()
        task.augmentations = replacement
        assert task.model.augmentations is replacement

    @pytest.mark.parametrize(
        'name',
        [
            'chesapeake_cvpr_prior_byol',
            'hyspecnet_byol',
            'seco_byol_1',
            'seco_byol_2',
            'ssl4eo_l_byol_1',
            'ssl4eo_l_byol_2',
            'ssl4eo_s12_byol_1',
            'ssl4eo_s12_byol_2',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        if name.startswith('seco'):
            monkeypatch.setattr(SeasonalContrastS2, '__len__', lambda self: 2)

        if name.startswith('ssl4eo_s12'):
            monkeypatch.setattr(SSL4EOS12, '__len__', lambda self: 2)

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
        with pytest.warns(UserWarning):
            BYOL(model='resnet18', in_channels=13, weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        match = 'num classes .* != num classes in pretrained model'
        with pytest.warns(UserWarning, match=match):
            BYOL(
                model=mocked_weights.meta['model'],
                weights=mocked_weights,
                in_channels=mocked_weights.meta['in_chans'],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        match = 'num classes .* != num classes in pretrained model'
        with pytest.warns(UserWarning, match=match):
            BYOL(
                model=mocked_weights.meta['model'],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta['in_chans'],
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        BYOL(
            model=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        BYOL(
            model=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )
