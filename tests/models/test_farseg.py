# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pytest
import torch

from torchgeo.models import FarSeg


class TestFarSeg:
    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize(
        "backbone,pretrained",
        [
            ("resnet18", True),
            ("resnet34", False),
            ("resnet50", True),
            ("resnet101", False),
        ],
    )
    def test_valid_backbone(self, backbone: str, pretrained: bool) -> None:
        model = FarSeg(classes=4, backbone=backbone, backbone_pretrained=pretrained)
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape == (2, 4, 128, 128)

    def test_invalid_backbone(self) -> None:
        match = "unknown backbone: anynet."
        with pytest.raises(ValueError, match=match):
            FarSeg(classes=4, backbone="anynet", backbone_pretrained=False)
