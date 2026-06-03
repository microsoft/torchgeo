# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Climplicit, Climplicit_Weights, climplicit


class TestClimplicit:
    @pytest.fixture(params=[*Climplicit_Weights])
    def weights(self, request: SubRequest) -> Climplicit_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Climplicit_Weights:
        weights = Climplicit_Weights.CHELSA
        path = tmp_path / f'{weights}.pth'
        model = Climplicit()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_climplicit(self) -> None:
        batch_size = 2
        model = Climplicit()
        coords = torch.randn(batch_size, 2)
        month = torch.ones(batch_size)
        out = model(coords, month)
        assert out.shape == torch.Size([batch_size, 256])

    def test_climplicit_no_month(self) -> None:
        batch_size = 2
        model = Climplicit()
        coords = torch.randn(batch_size, 2)
        out = model(coords)
        assert out.shape == torch.Size([batch_size, 1024])

    def test_climplicit_return_chelsa(self) -> None:
        batch_size = 2
        model = Climplicit(return_chelsa=True)
        coords = torch.randn(batch_size, 2)
        month = torch.ones(batch_size)
        out = model(coords, month)
        assert out.shape == torch.Size([batch_size, 11])

    def test_climplicit_return_chelsa_no_month(self) -> None:
        batch_size = 2
        model = Climplicit(return_chelsa=True)
        coords = torch.randn(batch_size, 2)
        out = model(coords)
        assert out.shape == torch.Size([batch_size, 44])

    def test_climplicit_no_weights(self) -> None:
        climplicit()

    def test_climplicit_weights(self, mocked_weights: Climplicit_Weights) -> None:
        climplicit(weights=mocked_weights)

    def test_transforms(self, weights: Climplicit_Weights) -> None:
        coords = torch.randn(2, 2)
        weights.transforms(coords)

    @pytest.mark.slow
    def test_climplicit_download(self, weights: Climplicit_Weights) -> None:
        climplicit(weights=weights)
