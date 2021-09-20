# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import FarSeg


class TestFarSeg:
    def test_in_channels(self) -> None:
        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet50", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        model(x)

        match = (
            "Unsupported in_channels = 5 for the pretrained model, "
            "only in_channels = 3 can be used currently"
        )
        with pytest.raises(ValueError, match=match):
            model = FarSeg(
                in_channels=5, classes=4, backbone="resnet50", backbone_pretrained=False
            )

    def test_classes(self) -> None:
        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet50", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape[1] == 4

    def test_output_size(self) -> None:
        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet50", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape[2] == 128 and y.shape[3] == 128

    def test_backbone(self) -> None:
        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet50", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape == (2, 4, 128, 128)

        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet18", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape == (2, 4, 128, 128)

        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet34", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape == (2, 4, 128, 128)

        model = FarSeg(
            in_channels=3, classes=4, backbone="resnet101", backbone_pretrained=False
        )
        x = torch.randn(2, 3, 128, 128)
        y = model(x)

        assert y.shape == (2, 4, 128, 128)

        match = "unknown backbone: anynet."
        with pytest.raises(ValueError, match=match):
            model = FarSeg(
                in_channels=3, classes=4, backbone="anynet", backbone_pretrained=False
            )
