# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import segmentation_models_pytorch as smp
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Unet_Weights, unet


class TestUnet:
    @pytest.fixture(params=[*Unet_Weights])
    def weights(self, request: SubRequest) -> Unet_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Unet_Weights:
        weights = Unet_Weights.SENTINEL2_2CLASS_FTW
        path = tmp_path / f'{weights}.pth'
        model = smp.Unet(
            in_channels=weights.meta['in_chans'],
            encoder_name=weights.meta['encoder'],
            encoder_weights=None,
            classes=weights.meta['num_classes'],
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_unet(self) -> None:
        unet()

    def test_unet_weights(self, mocked_weights: Unet_Weights) -> None:
        unet(weights=mocked_weights)

    def test_unet_weights_different_num_classes(
        self, mocked_weights: Unet_Weights
    ) -> None:
        unet(weights=mocked_weights, classes=20)

    def test_bands(self, weights: Unet_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Unet_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: Unet_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_unet_download(self, weights: Unet_Weights) -> None:
        unet(weights=weights)
