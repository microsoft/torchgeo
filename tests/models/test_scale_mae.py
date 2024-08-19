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

from torchgeo.models import ScaleMAELarge16_Weights, scalemae_large_patch16


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


class TestScaleMAE:
    @pytest.fixture(params=[*ScaleMAELarge16_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        model = scalemae_large_patch16()
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', load)
        return weights

    def test_scalemae(self) -> None:
        scalemae_large_patch16()

    def test_scalemae_forward_pass(self) -> None:
        model = scalemae_large_patch16(img_size=64, num_classes=2)
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert y.shape == (1, 2)

    def test_scalemae_weights(self, mocked_weights: WeightsEnum) -> None:
        scalemae_large_patch16(weights=mocked_weights)

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        mocked_weights.transforms(sample)

    def test_scalemae_weights_diff_image_size(
        self, mocked_weights: WeightsEnum
    ) -> None:
        scalemae_large_patch16(weights=mocked_weights, img_size=256)

    @pytest.mark.slow
    def test_scalemae_download(self, weights: WeightsEnum) -> None:
        scalemae_large_patch16(weights=weights)
