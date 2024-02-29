# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torchvision
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    DOFA,
    DOFABase16_Weights,
    dofa_base_patch16_224,
    dofa_huge_patch16_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
)


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


class TestDOFA:
    @pytest.mark.parametrize(
        "wavelengths",
        [
            # Gaofen
            [0.443, 0.565, 0.763, 0.765, 0.910],
            # NAIP
            [0.640, 0.560, 0.480],
            [0.480, 0.560, 0.640, 0.810],
            # Sentinel-1
            [5.405],
            [5.405, 5.405],
            # Sentinel-2
            [
                0.443,
                0.490,
                0.560,
                0.665,
                0.705,
                0.740,
                0.783,
                0.842,
                0.865,
                0.945,
                1.375,
                1.610,
                2.190,
            ],
        ],
    )
    def test_dofa(self, wavelengths: list[float]) -> None:
        batch_size = 2
        num_channels = len(wavelengths)
        num_classes = 10
        global_pool = num_channels % 2 == 0
        model = DOFA(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=num_classes,
            global_pool=global_pool,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # type: ignore[arg-type]
        )
        batch = torch.randn([batch_size, num_channels, 224, 224])
        out = model(batch, wavelengths)
        assert out.shape == torch.Size([batch_size, num_classes])


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
