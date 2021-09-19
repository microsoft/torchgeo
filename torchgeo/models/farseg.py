# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Foreground-Aware Relation Network (FarSeg) implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN
from typing import List, Tuple
import math
from collections import OrderedDict


class FarSeg(nn.Module):
    """Foreground-Aware Relation Network (FarSeg).

    'Foreground-Aware Relation Network for Geospatial Object Segmentation
    in High Spatial Resolution Remote Sensing Imagery', Zheng et al. (2020)
    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766.pdf
    @inproceedings{zheng2020foreground,
      title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
      author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={4096--4105},
      year={2020}
    }
    """

    def __init__(self,
                 in_channels: int,
                 backbone: str,
                 classes: int,
                 backbone_pretrained: bool = True
                 ) -> None:
        """Initializes the FarSeg.

        Args:
            in_channels: number of channels per input image
            backbone: name of ResNet backbone
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
                (default=True)
        """
        super().__init__()
        if backbone in ['resnet18', 'resnet34']:
            max_channels = 512
        elif backbone in ['resnet50', 'resnet101']:
            max_channels = 2048
        else:
            raise ValueError(f'unknown backbone: {backbone}.')
        self.backbone = getattr(resnet, backbone)(pretrained=backbone_pretrained)

        self.reset_in_channels_for_backbone(in_channels)

        self.fpn = FPN(in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)], out_channels=256)
        self.fsr = FSRelation(max_channels, [256] * 4, 256, True)
        self.decoder = LightWeightDecoder(256, 128, classes)

    def reset_in_channels_for_backbone(self, in_channels):
        if in_channels == 3:
            return

        self.backbone.add_module('conv1',
                                 nn.Conv2d(in_channels,
                                           64, kernel_size=7, stride=2, padding=3,
                                           bias=False))

    def backbone_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return c2, c3, c4, c5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone_forward(x)

        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        features = self.fpn(OrderedDict({f'c{i + 2}': features[i] for i in range(4)}))
        features = [v for k, v in features.items()]
        features = self.fsr(scene_embedding, features)

        logit = self.decoder(features)

        return logit


class FSRelation(nn.Module):
    """
    F-S Relation module in 'Foreground-Aware Relation Network for Geospatial Object Segmentation
    in High Spatial Resolution Remote Sensing Imagery', Zheng et al. (2020)
    """

    def __init__(self,
                 scene_embedding_channels: int,
                 in_channels_list: List[int],
                 out_channels: int,
                 scale_aware_proj: bool = True) -> None:
        """Initializes the FSRelation module.

        Args:
            scene_embedding_channels: number of scene embedding channels
            in_channels_list: a list of input channels
            out_channels: number of output channels
            scale_aware_proj: whether to use scale aware projection
                (default=True)
        """
        super(FSRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(scene_embedding_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(in_channels_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_embedding_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # [N, C, H, W]
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            # [N, C, 1, 1]
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class LightWeightDecoder(nn.Module):
    """
    Light Weight Decoder in 'Foreground-Aware Relation Network for Geospatial Object Segmentation
    in High Spatial Resolution Remote Sensing Imagery', Zheng et al. (2020)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_classes: int,
                 in_feature_output_strides: Tuple[int] = (4, 8, 16, 32),
                 out_feature_output_stride: int = 4,
                 ) -> None:
        """Initializes the LightWeightDecoder module.

        Args:
            in_channels: number of channels of input feature maps
            out_channels: number of channels of output feature maps
            num_classes: number of output segmentation classes
            in_feature_output_strides: output stride of input feature maps at different levels
                (default=(4, 8, 16, 32))
            out_feature_output_stride: output stride of output feature maps
                (default=4)
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feature_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

        self.classifier = nn.Sequential(
            nn.Conv2d(out_channels, num_classes, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        out_feat = self.classifier(out_feat)
        return out_feat
