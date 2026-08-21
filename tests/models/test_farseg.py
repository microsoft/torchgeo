# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import pytest
import torch

from torchgeo.models import FarSeg


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
