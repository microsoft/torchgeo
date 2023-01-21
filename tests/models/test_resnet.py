# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class TestResNet18:
    @pytest.fixture(params=[*ResNet18_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"])
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights, "url", path.as_uri())
        return weights

    def test_resnet(self) -> None:
        resnet18()

    def test_resnet_weights(self, mocked_weights: WeightsEnum) -> None:
        resnet18(weights=mocked_weights)

    @pytest.mark.slow
    def test_resnet_download(self, weights: WeightsEnum) -> None:
        resnet18(weights=weights)


class TestResNet50:
    @pytest.fixture(params=[*ResNet50_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model("resnet50", in_chans=weights.meta["in_chans"])
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights, "url", path.as_uri())
        return weights

    def test_resnet(self) -> None:
        resnet50()

    def test_resnet_weights(self, mocked_weights: WeightsEnum) -> None:
        resnet50(weights=mocked_weights)

    @pytest.mark.slow
    def test_resnet_download(self, weights: WeightsEnum) -> None:
        resnet50(weights=weights)
