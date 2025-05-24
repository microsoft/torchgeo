# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import segmentation_models_pytorch as smp
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Unet_Weights, unet


class TestUnet:
    @pytest.fixture(params=[*Unet_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Unet_Weights.SENTINEL2_2CLASS_FTW
        path = tmp_path / f'{weights}.pth'
        model = smp.Unet(
            in_channels=weights.meta['in_chans'],
            encoder_name=weights.meta['encoder'],
            encoder_weights=None,
            classes=weights.meta['num_classes'],
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_unet(self) -> None:
        unet()

    def test_unet_weights(self, mocked_weights: WeightsEnum) -> None:
        unet(weights=mocked_weights)

    def test_unet_weights_different_num_classes(
        self, mocked_weights: WeightsEnum
    ) -> None:
        unet(weights=mocked_weights, classes=20)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        if 'in_chans_transform' in weights.meta:
            c = weights.meta['in_chans_transforms']
        else:
            c = weights.meta['in_chans']

        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    @pytest.mark.slow
    def test_unet_download(self, weights: WeightsEnum) -> None:
        unet(weights=weights)
