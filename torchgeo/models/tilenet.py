# Copyright (c) TorchGeo Contributors.
# Licensed under the MIT License.

"""
TileNet encoder from Tile2Vec.

Reference:
Jean et al., Tile2Vec: Unsupervised Representation Learning
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = ["TileNet", "make_tilenet", "tilenet"]

_TILENET_NAIP_URL = (
    "https://huggingface.co/pgangapurwala/TileNet_Weights.NAIP_ALL_TILE2VEC/resolve/main/TileNet_Weights.NAIP_ALL_TILE2VEC.pth"
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, no_relu=False):
        super().__init__()
        self.no_relu = no_relu

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # extra conv3/bn3 branch (critical)
        self.conv3 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        if self.no_relu:
            return self.bn3(self.conv3(out))

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class TileNet(nn.Module):
    """TileNet encoder (Tile2Vec NAIP exact)."""

    name = "tilenet"
    embedding_dim = 512

    def __init__(self, in_channels=4, z_dim=512):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.layer5 = self._make_layer(
            z_dim, 2, stride=2, no_relu=True
        )

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, s in enumerate(strides):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=s,
                    no_relu=no_relu and i == num_blocks - 1,
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        return x.view(x.size(0), -1)


def make_tilenet(in_channels=4, z_dim=512):
    """Internal factory."""
    return TileNet(in_channels=in_channels, z_dim=z_dim)


def tilenet(pretrained: bool = False, **kwargs):
    """TileNet encoder (Tile2Vec NAIP).

    Args:
        pretrained: If True, load NAIP pretrained weights.
    """
    model = make_tilenet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            _TILENET_NAIP_URL,
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model