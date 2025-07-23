# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import ScaleMAELarge16_Weights, scalemae_large_patch16


class TestScaleMAE:
    @pytest.fixture(params=[*ScaleMAELarge16_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = ScaleMAELarge16_Weights.FMOW_RGB
        path = tmp_path / f'{weights}.pth'
        model = scalemae_large_patch16()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
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

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: WeightsEnum) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch._dynamo.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 224, 224, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    def test_scalemae_weights_diff_image_size(
        self, mocked_weights: WeightsEnum
    ) -> None:
        scalemae_large_patch16(weights=mocked_weights, img_size=256)

    @pytest.mark.slow
    def test_scalemae_download(self, weights: WeightsEnum) -> None:
        scalemae_large_patch16(weights=weights)
