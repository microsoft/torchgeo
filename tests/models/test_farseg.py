# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pytest
import torch

from torchgeo.models import FarSeg

BACKBONE = ["resnet18", "resnet34", "resnet50", "resnet101", "anynet"]


class TestFarSeg:
    @torch.no_grad()  # type: ignore[misc]
    def test_classes(self) -> None:
        model = FarSeg(classes=4, backbone="resnet50", backbone_pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape[1] == 4

    @torch.no_grad()  # type: ignore[misc]
    def test_output_size(self) -> None:
        model = FarSeg(classes=4, backbone="resnet50", backbone_pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape[2] == 128 and y.shape[3] == 128

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("backbone", BACKBONE)
    def test_backbone(self, backbone: str) -> None:
        if backbone == "anynet":
            match = "unknown backbone: anynet."
            with pytest.raises(ValueError, match=match):
                model = FarSeg(classes=4, backbone="anynet", backbone_pretrained=False)
        else:
            model = FarSeg(classes=4, backbone=backbone, backbone_pretrained=False)
            x = torch.randn(2, 3, 128, 128)
            y = model(x)

            assert y.shape == (2, 4, 128, 128)
