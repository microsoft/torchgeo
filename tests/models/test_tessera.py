# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Tessera, Tessera_Weights, tessera


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
    @pytest.fixture(params=[Tessera_Weights.TESSERA])
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
        tessera(weights=Tessera_Weights(mocked_weights))

    def test_transforms(self, weights: WeightsEnum) -> None:
        x = torch.randn(2, 10, 14)
        weights.transforms(x)

    @pytest.mark.slow
    def test_tessera_download(self, weights: WeightsEnum) -> None:
        tessera(weights=Tessera_Weights(weights))


class TestTesseraS2Encoder:
    @pytest.fixture(params=[Tessera_Weights.S2_ENCODER])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Tessera_Weights.S2_ENCODER
        path = tmp_path / f'{weights}.pth'
        model = tessera(model='s2')
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tessera_s2_encoder(self) -> None:
        model = tessera(model='s2')
        x = torch.randn(2, 10, 11)
        out = model(x)
        assert out.shape == torch.Size([2, 512])

    def test_tessera_s2_encoder_weights(self, mocked_weights: WeightsEnum) -> None:
        tessera(weights=Tessera_Weights(mocked_weights), model='s2')

    def test_transforms(self, weights: WeightsEnum) -> None:
        x = torch.randn(2, 10, 11)
        weights.transforms(x)

    @pytest.mark.slow
    def test_tessera_s2_download(self, weights: WeightsEnum) -> None:
        tessera(weights=Tessera_Weights(weights), model='s2')


class TestTesseraS1Encoder:
    @pytest.fixture(params=[Tessera_Weights.S1_ENCODER])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Tessera_Weights.S1_ENCODER
        path = tmp_path / f'{weights}.pth'
        model = tessera(model='s1')
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_tessera_s1_encoder(self) -> None:
        model = tessera(model='s1')
        x = torch.randn(2, 10, 3)
        out = model(x)
        assert out.shape == torch.Size([2, 512])

    def test_tessera_s1_encoder_weights(self, mocked_weights: WeightsEnum) -> None:
        tessera(weights=Tessera_Weights(mocked_weights), model='s1')

    def test_transforms(self, weights: WeightsEnum) -> None:
        x = torch.randn(2, 10, 3)
        weights.transforms(x)

    @pytest.mark.slow
    def test_tessera_s1_download(self, weights: WeightsEnum) -> None:
        tessera(weights=Tessera_Weights(weights), model='s1')
