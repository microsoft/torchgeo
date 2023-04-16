# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import pytest
import timm
import torch
import torchvision
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


class TestViTSmall16:
    @pytest.fixture(params=[*ViTSmall16_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model(
            weights.meta["model"], in_chans=weights.meta["in_chans"]
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, "url", str(path))
        except AttributeError:
            monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_vit(self) -> None:
        vit_small_patch16_224()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_small_patch16_224(weights=mocked_weights)

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta["in_chans"]
        sample = {"image": torch.arange(c * 4 * 4, dtype=torch.float).view(c, 4, 4)}
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_vit_download(self, weights: WeightsEnum) -> None:
        vit_small_patch16_224(weights=weights)
