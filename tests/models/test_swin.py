# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import pytest
import torch
import torchvision
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Swin_V2_B_Weights, swin_v2_b


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


class TestSwin_V2_B:
    @pytest.fixture(params=[*Swin_V2_B_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = torchvision.models.swin_v2_b()
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, "url", str(path))
        except AttributeError:
            monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_swin_v2_b(self) -> None:
        swin_v2_b()

    def test_swin_v2_b_weights(self, mocked_weights: WeightsEnum) -> None:
        swin_v2_b(weights=mocked_weights)

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta["in_chans"]
        sample = {
            "image": torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_swin_v2_b_download(self, weights: WeightsEnum) -> None:
        swin_v2_b(weights=weights)
