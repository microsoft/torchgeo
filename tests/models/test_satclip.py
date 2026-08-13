# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import SatCLIP, SatCLIP_Weights, satclip


class TestSatCLIP:
    @pytest.fixture(params=[*SatCLIP_Weights])
    def weights(self, request: SubRequest) -> SatCLIP_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> SatCLIP_Weights:
        weights = SatCLIP_Weights.SATCLIP_VIT16_L40
        path = tmp_path / f'{weights}.ckpt'
        model = SatCLIP(
            legendre_polys=weights.meta['legendre_polys'],
            capacity=weights.meta['capacity'],
            embed_dim=weights.meta['embed_dim'],
            num_hidden_layers=weights.meta['num_hidden_layers'],
        )
        state_dict = {
            f'model.location.{key}': value for key, value in model.state_dict().items()
        }
        torch.save({'state_dict': state_dict}, path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_satclip(self) -> None:
        model = satclip()
        coords = torch.tensor([[0.0, 0.0], [-122.0, 47.0], [139.7, 35.7]])
        embeddings = model(coords)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (3, 256)

    def test_satclip_custom_dimensions(self) -> None:
        model = satclip(
            legendre_polys=10, capacity=128, embed_dim=64, num_hidden_layers=1
        )
        coords = torch.zeros(2, 2)
        embeddings = model(coords)

        assert embeddings.shape == (2, 64)

    def test_satclip_coordinate_dtype(self) -> None:
        model = satclip(
            legendre_polys=10, capacity=128, embed_dim=64, num_hidden_layers=1
        )
        coords = torch.zeros(2, 2, dtype=torch.float64)
        embeddings = model(coords)

        assert embeddings.dtype == torch.float32
        assert embeddings.shape == (2, 64)

    def test_satclip_pole_clamping(self) -> None:
        model = satclip(
            legendre_polys=10, capacity=64, embed_dim=32, num_hidden_layers=1
        )
        coords = torch.tensor([[0.0, 90.0], [0.0, -90.0]])
        embeddings = model(coords)

        assert torch.isfinite(embeddings).all()

    @pytest.mark.parametrize(
        ('legendre_polys', 'capacity', 'embed_dim', 'num_hidden_layers'),
        [(0, 512, 256, 2), (40, 0, 256, 2), (40, 512, 0, 2), (40, 512, 256, 0)],
    )
    def test_satclip_invalid_dimensions(
        self, legendre_polys: int, capacity: int, embed_dim: int, num_hidden_layers: int
    ) -> None:
        with pytest.raises(AssertionError):
            SatCLIP(legendre_polys, capacity, embed_dim, num_hidden_layers)

    def test_satclip_weights(self, mocked_weights: SatCLIP_Weights) -> None:
        model = satclip(weights=mocked_weights)
        coords = torch.tensor([[0.0, 0.0]])
        embeddings = model(coords)

        assert not model.training
        assert embeddings.shape == (1, mocked_weights.meta['embed_dim'])

    def test_transforms(self, weights: SatCLIP_Weights) -> None:
        coords = torch.tensor([[0.0, 0.0]])
        weights.transforms(coords)

    @pytest.mark.slow
    def test_satclip_download(self, weights: SatCLIP_Weights) -> None:
        satclip(weights=weights)
