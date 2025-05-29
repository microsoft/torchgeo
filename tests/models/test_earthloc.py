# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import EarthLoc_Weights, earthloc


class TestEarthLoc:
    @pytest.fixture(params=[*EarthLoc_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = EarthLoc_Weights.SENTINEL2_RESNET50
        path = tmp_path / f'{weights}.pth'
        kwargs = {
            'in_channels': weights.meta['in_chans'],
            'image_size': weights.meta['image_size'],
            'desc_dim': weights.meta['desc_dim'],
            'backbone': weights.meta['encoder'],
            'pretrained': False,
        }
        model = earthloc(**kwargs)
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_earthloc(self) -> None:
        earthloc()

    def test_earthloc_weights(self, mocked_weights: WeightsEnum) -> None:
        earthloc(weights=mocked_weights)

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
    def test_earthloc_download(self, weights: WeightsEnum) -> None:
        earthloc(weights=weights)
