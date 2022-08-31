# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools

import pytest
import torch
import torch.nn as nn
from torch.nn.modules import Module

from torchgeo.models import ChangeMixin, ChangeStar, ChangeStarFarSeg

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

BACKBONE = ["resnet18", "resnet34", "resnet50", "resnet101"]
IN_CHANNELS = [64, 128]
INNNR_CHANNELS = [16, 32, 64]
NC = [1, 2, 4]
SF = [4.0, 8.0, 1.0]


class TestChangeStar:
    @torch.no_grad()  # type: ignore[misc]
    def test_changestar_farseg_classes(self) -> None:
        model = ChangeStarFarSeg(
            classes=4, backbone="resnet50", backbone_pretrained=False
        )
        x = torch.randn(2, 2, 3, 128, 128)
        y = model(x)

        assert y["bi_seg_logit"].shape[2] == 4

    @torch.no_grad()  # type: ignore[misc]
    def test_changestar_farseg_output_size(self) -> None:
        model = ChangeStarFarSeg(
            classes=4, backbone="resnet50", backbone_pretrained=False
        )
        model.eval()
        x = torch.randn(2, 2, 3, 128, 128)
        y = model(x)

        assert y["bi_seg_logit"].shape[3] == 128 and y["bi_seg_logit"].shape[4] == 128
        assert y["change_prob"].shape[2] == 128 and y["change_prob"].shape[3] == 128

        model.train()
        y = model(x)

        assert y["bi_seg_logit"].shape[3] == 128 and y["bi_seg_logit"].shape[4] == 128
        assert y["bi_change_logit"].shape[3] == 128
        assert y["bi_change_logit"].shape[4] == 128

    @pytest.mark.parametrize("backbone", BACKBONE)
    def test_valid_changestar_farseg_backbone(self, backbone: str) -> None:
        ChangeStarFarSeg(classes=4, backbone=backbone, backbone_pretrained=False)

    def test_invalid_changestar_farseg_backbone(self) -> None:
        match = "unknown backbone: anynet."
        with pytest.raises(ValueError, match=match):
            ChangeStarFarSeg(classes=4, backbone="anynet", backbone_pretrained=False)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize(
        "inc,innerc,nc,sf", list(itertools.product(IN_CHANNELS, INNNR_CHANNELS, NC, SF))
    )
    def test_changemixin_output_size(
        self, inc: int, innerc: int, nc: int, sf: float
    ) -> None:
        m = ChangeMixin(
            in_channels=inc, inner_channels=innerc, num_convs=nc, scale_factor=sf
        )

        y = m(torch.rand(3, 2, inc // 2, 32, 32))
        assert y[0].shape == y[1].shape
        assert y[0].shape == (3, 1, int(32 * sf), int(32 * sf))

    @torch.no_grad()  # type: ignore[misc]
    def test_changestar(self) -> None:
        dense_feature_extractor = nn.modules.Sequential(
            nn.modules.Conv2d(3, 32, 3, 1, 1),
            nn.modules.BatchNorm2d(32),
            nn.modules.ReLU(),
            nn.modules.MaxPool2d(3, 2, 1),
        )

        seg_classifier = nn.modules.Sequential(
            nn.modules.Conv2d(32, 2, 3, 1, 1),
            nn.modules.UpsamplingBilinear2d(scale_factor=2.0),
        )

        m = ChangeStar(
            dense_feature_extractor,
            seg_classifier,
            ChangeMixin(
                in_channels=32 * 2, inner_channels=16, num_convs=4, scale_factor=2.0
            ),
        )
        m.eval()

        y = m(torch.rand(3, 2, 3, 64, 64))
        assert y["bi_seg_logit"].shape == (3, 2, 2, 64, 64)
        assert y["change_prob"].shape == (3, 1, 64, 64)

    @torch.no_grad()  # type: ignore[misc]
    def test_changestar_invalid_inference_mode(self) -> None:
        dense_feature_extractor = nn.modules.Sequential(
            nn.modules.Conv2d(3, 32, 3, 1, 1),
            nn.modules.BatchNorm2d(32),
            nn.modules.ReLU(),
            nn.modules.MaxPool2d(3, 2, 1),
        )

        seg_classifier = nn.modules.Sequential(
            nn.modules.Conv2d(32, 2, 3, 1, 1),
            nn.modules.UpsamplingBilinear2d(scale_factor=2.0),
        )

        match = "Unknown inference_mode: random"
        with pytest.raises(ValueError, match=match):
            ChangeStar(
                dense_feature_extractor,
                seg_classifier,
                ChangeMixin(
                    in_channels=32 * 2, inner_channels=16, num_convs=4, scale_factor=2.0
                ),
                inference_mode="random",
            )

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("inference_mode", ["t1t2", "t2t1", "mean"])
    def test_changestar_inference_output_size(self, inference_mode: str) -> None:
        dense_feature_extractor = nn.modules.Sequential(
            nn.modules.Conv2d(3, 32, 3, 1, 1),
            nn.modules.BatchNorm2d(32),
            nn.modules.ReLU(),
            nn.modules.MaxPool2d(3, 2, 1),
        )
        CLASSES = 2
        seg_classifier = nn.modules.Sequential(
            nn.modules.Conv2d(32, CLASSES, 3, 1, 1),
            nn.modules.UpsamplingBilinear2d(scale_factor=2.0),
        )

        m = ChangeStar(
            dense_feature_extractor,
            seg_classifier,
            ChangeMixin(
                in_channels=32 * 2, inner_channels=16, num_convs=4, scale_factor=2.0
            ),
            inference_mode=inference_mode,
        )
        m.eval()

        x = torch.randn(2, 2, 3, 128, 128)
        y = m(x)

        assert y["bi_seg_logit"].shape == (2, 2, CLASSES, 128, 128)
        assert y["change_prob"].shape == (2, 1, 128, 128)
