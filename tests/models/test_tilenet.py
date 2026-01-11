# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import TileNet_Weights, tilenet
from torchgeo.models.tilenet import BasicBlock, make_tilenet


class TestTileNet:
    """Tests for TileNet (Tile2Vec) model."""

    @pytest.fixture(params=[*TileNet_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        """Return all available TileNet weights."""
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        """Mock TileNet weights download."""
        weights = TileNet_Weights.NAIP
        path = tmp_path / f'{weights}.pth'

        model = tilenet(in_channels=weights.meta['in_chans'])
        torch.save(model.state_dict(), path)

        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tilenet(self) -> None:
        """Test TileNet construction without weights."""
        model = tilenet()
        x = torch.randn(1, 4, 50, 50)
        y = model(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, 512)

    def test_tilenet_weights(self, mocked_weights: WeightsEnum) -> None:
        """Test TileNet with pretrained weights."""
        model = tilenet(weights=mocked_weights)
        x = torch.randn(1, mocked_weights.meta['in_chans'], 50, 50)
        y = model(x)

        assert y.shape[1] == 512

    def test_make_tilenet(self) -> None:
        """Test internal TileNet factory."""
        model = make_tilenet(in_channels=4, z_dim=512)
        x = torch.randn(1, 4, 50, 50)
        y = model(x)

        assert y.shape == (1, 512)

    def test_basicblock_no_relu(self) -> None:
        """Test BasicBlock with no_relu=True branch."""
        block = BasicBlock(in_planes=64, planes=64, no_relu=True)
        x = torch.randn(1, 64, 10, 10)
        y = block(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape

    def test_bands(self, weights: WeightsEnum) -> None:
        """Test bands metadata consistency."""
        assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        """Test weight transforms."""
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 50 * 50, dtype=torch.float).view(c, 50, 50)
        }
        weights.transforms(sample)

    @pytest.mark.slow
    def test_tilenet_download(self, weights: WeightsEnum) -> None:
        """Test real weight download."""
        tilenet(weights=weights)

    @pytest.mark.optional
    def test_tilenet_custom_dimensions(self) -> None:
        """Optional test: custom input channels and embedding size."""
        model = make_tilenet(in_channels=6, z_dim=256)
        x = torch.randn(1, 6, 64, 64)
        y = model(x)

        assert y.shape == (1, 256)