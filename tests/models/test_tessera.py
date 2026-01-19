# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Tessera, Tessera_Weights, tessera


class TestTessera:
    @pytest.fixture(params=[*Tessera_Weights])
    def weights(self, request: SubRequest) -> Tessera_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Tessera_Weights:
        weights = Tessera_Weights.TESSERA
        path = tmp_path / f'{weights}.pth'
        model = Tessera()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    @pytest.fixture
    def mocked_weights_s2(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Tessera_Weights:
        weights = Tessera_Weights.TESSERA_SENTINEL2_ENCODER
        path = tmp_path / f'{weights}.pth'
        model = Tessera()
        torch.save(model.s2_backbone.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    @pytest.fixture
    def mocked_weights_s1(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Tessera_Weights:
        weights = Tessera_Weights.TESSERA_SENTINEL1_ENCODER
        path = tmp_path / f'{weights}.pth'
        model = Tessera()
        torch.save(model.s1_backbone.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tessera(self) -> None:
        batch_size = 2
        seq_len = 10
        model = Tessera()
        x = torch.randn(batch_size, seq_len, 14)
        out = model(x)
        assert out.shape == torch.Size([batch_size, model.embed_dim])

    def test_tessera_no_weights(self) -> None:
        tessera()

    def test_tessera_custom_embed_dim(self) -> None:
        batch_size = 2
        seq_len = 5
        embed_dim = 64
        model = Tessera(embed_dim=embed_dim)
        x = torch.randn(batch_size, seq_len, 14)
        out = model(x)
        assert out.shape == torch.Size([batch_size, embed_dim])

    def test_tessera_invalid_input(self) -> None:
        model = Tessera()
        x = torch.randn(2, 10, 10)
        with pytest.raises(AssertionError, match='Expected 14 channels'):
            model(x)

    def test_tessera_weights(self, mocked_weights: Tessera_Weights) -> None:
        tessera(weights=mocked_weights)

    def test_tessera_s2_encoder_weights(
        self, mocked_weights_s2: Tessera_Weights
    ) -> None:
        model = tessera(weights=mocked_weights_s2)
        x = torch.randn(2, 10, 11)
        out = model(x)
        assert out.shape == torch.Size([2, 512])

    def test_tessera_s1_encoder_weights(
        self, mocked_weights_s1: Tessera_Weights
    ) -> None:
        model = tessera(weights=mocked_weights_s1)
        x = torch.randn(2, 10, 3)
        out = model(x)
        assert out.shape == torch.Size([2, 512])

    def test_bands(self, weights: Tessera_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Tessera_Weights) -> None:
        in_chans = weights.meta['in_chans']
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, in_chans)
        weights.transforms(x)

    def test_transforms_2d(self, weights: Tessera_Weights) -> None:
        in_chans = weights.meta['in_chans']
        seq_len = 10
        x = torch.randn(seq_len, in_chans)
        weights.transforms(x)

    def test_tessera_invalid_weights(self) -> None:
        with pytest.raises(ValueError, match='Unsupported weights'):
            # Create a mock invalid weight by using a non-existent enum value
            # This tests the else branch in tessera() function
            tessera(weights='invalid')  # type: ignore[arg-type]

    @pytest.mark.slow
    def test_tessera_download(self, weights: Tessera_Weights) -> None:
        tessera(weights=weights)
