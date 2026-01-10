# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import TileNet_Weights, tilenet


class TestTileNet:
    @pytest.fixture(params=[*TileNet_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = TileNet_Weights.NAIP
        path = tmp_path / f'{weights}.pth'

        # Create dummy TileNet checkpoint
        model = tilenet(in_channels=weights.meta['in_chans'])
        torch.save(model.state_dict(), path)

        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tilenet(self) -> None:
        tilenet()

    def test_tilenet_weights(self, mocked_weights: WeightsEnum) -> None:
        tilenet(weights=mocked_weights)

    def test_bands(self, weights: WeightsEnum) -> None:
        assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {'image': torch.arange(c * 50 * 50, dtype=torch.float).view(c, 50, 50)}
        weights.transforms(sample)

    @pytest.mark.slow
    def test_tilenet_download(self, weights: WeightsEnum) -> None:
        tilenet(weights=weights)
