# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import pytest
import torch

from torchgeo.models import FarSeg


class TestFarSeg:
    @torch.no_grad()
    @pytest.mark.parametrize(
        'backbone', ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    )
    def test_valid_backbone(self, backbone: str) -> None:
        model = FarSeg(classes=4, backbone=backbone)
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape == (2, 4, 128, 128)

    def test_invalid_backbone(self) -> None:
        match = 'unknown backbone: anynet.'
        with pytest.raises(ValueError, match=match):
            FarSeg(classes=4, backbone='anynet')
