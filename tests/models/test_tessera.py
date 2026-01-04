# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    Tessera,
    Tessera_S1_Encoder_Weights,
    Tessera_S2_Encoder_Weights,
    Tessera_Weights,
    tessera,
    tessera_s1_encoder,
    tessera_s2_encoder,
)


class TestTessera:
    def test_tessera(self) -> None:
        batch_size = 2
        seq_len = 10
        model = Tessera()
        x = torch.randn(batch_size, seq_len, 14)
        out = model(x)
        assert out.shape == torch.Size([batch_size, model.embed_dim])

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


class TestTesseraWeights:
    @pytest.fixture(params=[*Tessera_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Tessera_Weights.TESSERA
        path = tmp_path / f'{weights}.pth'
        model = tessera()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tessera(self) -> None:
        tessera()

    def test_tessera_weights(self, mocked_weights: WeightsEnum) -> None:
        tessera(weights=mocked_weights)

    def test_transforms(self, weights: WeightsEnum) -> None:
        x = torch.randn(2, 10, 14)
        weights.transforms(x)

    @pytest.mark.slow
    def test_tessera_download(self, weights: WeightsEnum) -> None:
        tessera(weights=weights)


class TestTesseraS2EncoderWeights:
    @pytest.fixture(params=[*Tessera_S2_Encoder_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Tessera_S2_Encoder_Weights.TESSERA
        path = tmp_path / f'{weights}.pth'
        model = tessera_s2_encoder()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tessera_s2_encoder(self) -> None:
        model = tessera_s2_encoder()
        x = torch.randn(2, 10, 11)
        out = model(x)
        assert out.shape == torch.Size([2, 512])

    def test_tessera_s2_encoder_weights(self, mocked_weights: WeightsEnum) -> None:
        tessera_s2_encoder(weights=mocked_weights)

    def test_transforms(self, weights: WeightsEnum) -> None:
        x = torch.randn(2, 10, 11)
        weights.transforms(x)

    @pytest.mark.slow
    def test_tessera_s2_download(self, weights: WeightsEnum) -> None:
        tessera_s2_encoder(weights=weights)


class TestTesseraS1EncoderWeights:
    @pytest.fixture(params=[*Tessera_S1_Encoder_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Tessera_S1_Encoder_Weights.TESSERA
        path = tmp_path / f'{weights}.pth'
        model = tessera_s1_encoder()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tessera_s1_encoder(self) -> None:
        model = tessera_s1_encoder()
        x = torch.randn(2, 10, 3)
        out = model(x)
        assert out.shape == torch.Size([2, 512])

    def test_tessera_s1_encoder_weights(self, mocked_weights: WeightsEnum) -> None:
        tessera_s1_encoder(weights=mocked_weights)

    def test_transforms(self, weights: WeightsEnum) -> None:
        x = torch.randn(2, 10, 3)
        weights.transforms(x)

    @pytest.mark.slow
    def test_tessera_s1_download(self, weights: WeightsEnum) -> None:
        tessera_s1_encoder(weights=weights)
