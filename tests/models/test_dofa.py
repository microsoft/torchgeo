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

from torchgeo.models import (
    DOFABase16_Weights,
    dofa_base_patch16_224,
    dofa_huge_patch16_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
)


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


class TestDOFASmall16:
    def test_dofa(self) -> None:
        dofa_small_patch16_224()


class TestDOFABase16:
    @pytest.fixture(params=[*DOFABase16_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = dofa_base_patch16_224()
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, "url", str(path))
        except AttributeError:
            monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_dofa(self) -> None:
        dofa_base_patch16_224()

    def test_dofa_weights(self, mocked_weights: WeightsEnum) -> None:
        dofa_base_patch16_224(weights=mocked_weights)

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = 4
        sample = {
            "image": torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_dofa_download(self, weights: WeightsEnum) -> None:
        dofa_base_patch16_224(weights=weights)


class TestDOFALarge16:
    def test_dofa(self) -> None:
        dofa_large_patch16_224()


class TestDOFAHuge16:
    def test_dofa(self) -> None:
        dofa_huge_patch16_224()
