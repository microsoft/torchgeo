# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torchvision.models._api import Weights

from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224


class TestViTSmall16:
    @pytest.fixture(scope="function", params=[*ViTSmall16_Weights])
    def weights(self, request: SubRequest) -> Weights:
        return request.param

    @pytest.fixture(scope="function")
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: Weights
    ) -> Weights:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model(
            weights.meta["model"], in_chans=weights.meta["in_chans"]
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights, "url", path.as_uri())
        return weights

    def test_vit(self) -> None:
        vit_small_patch16_224()

    def test_vit_weights(self, mocked_weights: Weights) -> None:
        vit_small_patch16_224(weights=mocked_weights)

    @pytest.mark.slow
    def test_vit_download(self, weights: Weights) -> None:
        vit_small_patch16_224(weights=weights)
