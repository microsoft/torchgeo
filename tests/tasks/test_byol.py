# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable
from pathlib import Path

import pytest
import timm
import torch
from pytest import MonkeyPatch
from torch import nn
from torchvision.models import resnet18
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import SSL4EOS12, ChesapeakeCVPR, SeasonalContrastS2
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
        BYOLModule(model, augment_fn=augment_fn)


class TestBYOL:
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
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        fast_dev_run: bool,
        test_config: Callable[[str], str],
    ) -> None:
        config = test_config(name + '.yaml')

        if name.startswith('seco'):
            monkeypatch.setattr(SeasonalContrastS2, '__len__', lambda self: 2)

        if name.startswith('chesapeake_cvpr'):
            # Tell ChesapeakeCVPR that only the fixture tile is available so it
            # doesn't try to (re-)extract the archives on every test run.
            # Without this, parallel test workers can extract into the same
            # shared tests/data directory at once and corrupt each other's reads.
            monkeypatch.setattr(
                ChesapeakeCVPR,
                '_files',
                {
                    'base': (
                        'de_1m_2013_extended-debuffered-test_tiles',
                        'spatial_index.geojson',
                    ),
                    'prior_extension': (
                        'de_1m_2013_extended-debuffered-test_tiles/m_3807504_ne_18_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif',
                    ),
                },
            )

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
