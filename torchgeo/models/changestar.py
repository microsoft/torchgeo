# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""ChangeStar implementations."""

import math
from collections import OrderedDict
from typing import List, cast, Dict

import torch.nn.functional as F
import torch
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
from .farseg import FarSeg


class ChangeMixin(Module):
    def __init__(self,
                 in_channels: int = 128,
                 inner_channels: int = 16,
                 num_convs: int = 4,
                 out_channels=1,
                 scale_factor: float = 4.
                 ):
        super(ChangeMixin, self).__init__()

        layers = [Sequential(
            Conv2d(in_channels, inner_channels, 3, 1, 1),
            ReLU(True),
            BatchNorm2d(inner_channels),
        )]
        layers += [Sequential(
            Conv2d(inner_channels, inner_channels, 3, 1, 1),
            ReLU(True),
            BatchNorm2d(inner_channels),
        ) for _ in range(num_convs - 1)]

        cls_layer = Conv2d(inner_channels, out_channels, 3, 1, 1)

        layers.append(cls_layer)
        layers.append(UpsamplingBilinear2d(scale_factor=scale_factor))

        self.convs = Sequential(*layers)

    def forward(self, x):

        t1t2, t2t1 = torch.cat([t1, t2], dim=1), torch.cat([t2, t1], dim=1)
        c1221 = self.convs(torch.cat([t1t2, t2t1], dim=0))
        c12, c21 = torch.split(c1221, t1t2.size(0), dim=0)


class ChangeStar(Module):
    def __init__(self,
                 dense_feature_extractor: Module) -> None:
        super(ChangeStar, self).__init__()  # type: ignore[no-untyped-call]
        self.dense_feature_extractor = dense_feature_extractor
        self.changemixin = ChangeMixin()

    def forward(self, x) -> Dict[str, Tensor]:
        bi_features = self.dense_feature_extractor(x)
        bi_seg_logit, change_logit = self.changemixin(bi_features)

        results = {
            'bi_seg_logit': bi_seg_logit,
            'change_logit': change_logit
        }

        return results


class ChangeStarFarSeg(ChangeStar):
    def __init__(self) -> None:
        model = FarSeg()
        model.decoder.classifier = Identity()
        super(ChangeStarFarSeg, self).__init__(
            dense_feature_extractor=model
        )  # type: ignore[no-untyped-call]
