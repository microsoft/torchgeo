# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Foreground-Aware Relation Network (FarSeg) implementations."""

import math
from collections import OrderedDict
from typing import List, cast

import torch.nn.functional as F
import torchvision
from packaging.version import parse
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    Identity,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    UpsamplingBilinear2d,
)
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"
ModuleList.__module__ = "nn.ModuleList"
Sequential.__module__ = "nn.Sequential"
Conv2d.__module__ = "nn.Conv2d"
BatchNorm2d.__module__ = "nn.BatchNorm2d"
ReLU.__module__ = "nn.ReLU"
UpsamplingBilinear2d.__module__ = "nn.UpsamplingBilinear2d"
Sigmoid.__module__ = "nn.Sigmoid"
Identity.__module__ = "nn.Identity"


class FarSeg(Module):
    """Foreground-Aware Relation Network (FarSeg).

    This model can be used for binary- or multi-class object segmentation, such as
    building, road, ship, and airplane segmentation. It can be also extended as a change
    detection model. It features a foreground-scene relation module to model the
    relation between scene embedding, object context, and object feature, thus improving
    the discrimination of object feature representation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766.pdf
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        classes: int = 16,
        backbone_pretrained: bool = True,
    ) -> None:
        """Initialize a new FarSeg model.

        Args:
            backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
                "resnet50", "resnet101"]
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        super().__init__()
        if backbone in ["resnet18", "resnet34"]:
            max_channels = 512
        elif backbone in ["resnet50", "resnet101"]:
            max_channels = 2048
        else:
            raise ValueError(f"unknown backbone: {backbone}.")
        kwargs = {}
        if parse(torchvision.__version__) >= parse("0.12"):
            if backbone_pretrained:
                kwargs = {
                    "weights": getattr(
                        torchvision.models, f"ResNet{backbone[6:]}_Weights"
                    ).DEFAULT
                }
            else:
                kwargs = {"weights": None}
        else:
            kwargs = {"pretrained": backbone_pretrained}

        self.backbone = getattr(resnet, backbone)(**kwargs)

        self.fpn = FPN(
            in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)],
            out_channels=256,
        )
        self.fsr = _FSRelation(max_channels, [256] * 4, 256)
        self.decoder = _LightWeightDecoder(256, 128, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            output prediction
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]

        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        fpn_features = self.fpn(
            OrderedDict({f"c{i + 2}": features[i] for i in range(4)})
        )
        features = [v for k, v in fpn_features.items()]
        features = self.fsr(scene_embedding, features)

        logit = self.decoder(features)

        return cast(Tensor, logit)


class _FSRelation(Module):
    """F-S Relation module."""

    def __init__(
        self,
        scene_embedding_channels: int,
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        """Initialize the _FSRelation module.

        Args:
            scene_embedding_channels: number of scene embedding channels
            in_channels_list: a list of input channels
            out_channels: number of output channels
        """
        super().__init__()

        self.scene_encoder = ModuleList(
            [
                Sequential(
                    Conv2d(scene_embedding_channels, out_channels, 1),
                    ReLU(True),
                    Conv2d(out_channels, out_channels, 1),
                )
                for _ in range(len(in_channels_list))
            ]
        )

        self.content_encoders = ModuleList()
        self.feature_reencoders = ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1), BatchNorm2d(out_channels), ReLU(True)
                )
            )
            self.feature_reencoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1), BatchNorm2d(out_channels), ReLU(True)
                )
            )

        self.normalizer = Sigmoid()

    def forward(self, scene_feature: Tensor, features: List[Tensor]) -> List[Tensor]:
        """Forward pass of the model."""
        # [N, C, H, W]
        content_feats = [
            c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)
        ]
        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats)
        ]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class _LightWeightDecoder(Module):
    """Light Weight Decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        in_feature_output_strides: List[int] = [4, 8, 16, 32],
        out_feature_output_stride: int = 4,
    ) -> None:
        """Initialize the _LightWeightDecoder module.

        Args:
            in_channels: number of channels of input feature maps
            out_channels: number of channels of output feature maps
            num_classes: number of output segmentation classes
            in_feature_output_strides: output stride of input feature maps at different
                levels
            out_feature_output_stride: output stride of output feature maps
        """
        super().__init__()

        self.blocks = ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feature_output_stride))
            )
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                Sequential(
                    *[
                        Sequential(
                            Conv2d(
                                in_channels if idx == 0 else out_channels,
                                out_channels,
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            BatchNorm2d(out_channels),
                            ReLU(inplace=True),
                            UpsamplingBilinear2d(scale_factor=2)
                            if num_upsample != 0
                            else Identity(),
                        )
                        for idx in range(num_layers)
                    ]
                )
            )

        self.classifier = Sequential(
            Conv2d(out_channels, num_classes, 3, 1, 1),
            UpsamplingBilinear2d(scale_factor=4),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward pass of the model."""
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        out_feat = self.classifier(out_feat)

        return cast(Tensor, out_feat)
