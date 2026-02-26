# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import TileNet_Weights, tilenet


class TestTileNet:
    """Tests for TileNet (Tile2Vec) model."""

    @pytest.fixture(params=[*TileNet_Weights])
    def weights(self, request: SubRequest) -> TileNet_Weights:
        """Return all available TileNet weights."""
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> TileNet_Weights:
        """Mock TileNet weights download."""
        weights = TileNet_Weights.NAIP_ALL_TILE2VEC
        path = tmp_path / f'{weights}.pth'

        model = tilenet(in_channels=weights.meta['in_chans'])
        torch.save(model.state_dict(), path)

        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tilenet(self) -> None:
        """Test TileNet construction and forward pass."""
        model = tilenet()
        x = torch.randn(1, 4, 50, 50)
        y = model(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, 512)

    def test_tilenet_custom_dimensions(self) -> None:
        """Test TileNet with custom embedding dimension."""
        model = tilenet(in_channels=4, z_dim=256)
        x = torch.randn(1, 4, 50, 50)
        y = model(x)

        assert y.shape == (1, 256)

    def test_tilenet_weights(self, mocked_weights: TileNet_Weights) -> None:
        """Test TileNet with pretrained weights."""
        model = tilenet(weights=mocked_weights)
        x = torch.randn(1, mocked_weights.meta['in_chans'], 50, 50)
        y = model(x)

        assert y.shape == (1, 512)

    def test_bands(self, weights: TileNet_Weights) -> None:
        """Test bands metadata consistency."""
        assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: TileNet_Weights) -> None:
        """Test that transforms run without error."""
        c = weights.meta['in_chans']
        sample = {'image': torch.arange(c * 50 * 50, dtype=torch.float).view(c, 50, 50)}
        weights.transforms(sample)

    @pytest.mark.slow
    def test_tilenet_download(self, weights: TileNet_Weights) -> None:
        """Test real weight download."""
        tilenet(weights=weights)
