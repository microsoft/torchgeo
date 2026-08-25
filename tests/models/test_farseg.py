# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


from pathlib import Path

import pytest
import torch
from pytest import MonkeyPatch
from timm import create_model

from torchgeo.models import FarSeg, ResNet50_Weights


class TestFarSeg:
    @torch.inference_mode()
    @pytest.mark.parametrize(
        'backbone', ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    )
    def test_valid_backbone(self, backbone: str) -> None:
        model = FarSeg(classes=4, backbone=backbone)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)

        assert y.shape == (2, 4, 32, 32)

    def test_invalid_backbone(self) -> None:
        match = 'unknown backbone: anynet.'
        with pytest.raises(ValueError, match=match):
            FarSeg(classes=4, backbone='anynet')

    @torch.inference_mode()
    def test_torchgeo_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> None:
        weights = ResNet50_Weights.LANDSAT_TM_TOA_MOCO
        path = tmp_path / f'{weights}.pth'
        model = create_model('resnet50', in_chans=weights.meta['in_chans'])
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))

        model = FarSeg(classes=4, backbone='resnet50', backbone_weights=weights)
        x = torch.randn(2, weights.meta['in_chans'], 32, 32)
        y = model(x)

        assert y.shape == (2, 4, 32, 32)

    def test_mismatched_torchgeo_weights(self) -> None:
        match = 'backbone weights are for resnet50, not resnet18.'
        with pytest.raises(ValueError, match=match):
            FarSeg(
                backbone='resnet18',
                backbone_weights=ResNet50_Weights.LANDSAT_TM_TOA_MOCO,
            )
