# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Aurora_Weights, aurora_swin_unet

pytest.importorskip('microsoft-aurora', minversion='1.7.0')


class TestAurora:
    @pytest.fixture(params=[*Aurora_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Aurora_Weights.HRES_T0_PRETRAINED_SMALL_AURORA
        path = tmp_path / f'{weights}.pth'
        model = aurora_swin_unet(weights=weights)
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_aurora_swin_unet(self) -> None:
        aurora_swin_unet()

    def test_aurora_swin_unet_weights(self, mocked_weights: WeightsEnum) -> None:
        aurora_swin_unet(weights=mocked_weights)

    def test_aurora_swin_unet_weights_different_num_classes(
        self, mocked_weights: WeightsEnum
    ) -> None:
        aurora_swin_unet(weights=mocked_weights, classes=20)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    @pytest.mark.slow
    def test_aurora_swin_unet_download(self, weights: WeightsEnum) -> None:
        aurora_swin_unet(weights=weights)
