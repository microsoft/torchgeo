# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/ermongroup/tile2vec. Copyright (c) 2024 Ermon Group

"""TileNet encoder from Tile2Vec.

Reference:
Jean et al., Tile2Vec: Unsupervised Representation Learning
"""

from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch import nn
from torchvision.models._api import Weights, WeightsEnum


class TileNet_Weights(WeightsEnum):
    """TileNet (Tile2Vec) weights.

    NAIP-pretrained Tile2Vec encoder.

    .. versionadded:: 0.9
    """

    NAIP_ALL_TILE2VEC = Weights(
        url=(
            'https://hf.co/pgangapurwala/TileNet_Weights.NAIP_ALL_TILE2VEC/resolve/af12210f5c130af76579ce8ec5e7036c1551ba25/TileNet_Weights.NAIP_ALL_TILE2VEC.pth'
        ),
        transforms=T.Normalize(mean=[0], std=[255], inplace=True),
        meta={
            'dataset': 'NAIP',
            'in_chans': 4,
            'model': 'tilenet',
            'ssl_method': 'tile2vec',
            'publication': 'https://arxiv.org/abs/1805.02855',
            'repo': 'https://github.com/ermongroup/tile2vec',
            'bands': ('R', 'G', 'B', 'NIR'),
        },
    )


class BasicBlock(nn.Module):
    """Tile2Vec residual block with extra conv3 branch."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Initialize a BasicBlock.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Convolution stride.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # extra conv3/bn3
        self.conv3 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class TileNet(nn.Module):
    """TileNet encoder.

    versionadded:: 0.9
    """

    def __init__(self, in_channels: int = 4, z_dim: int = 512) -> None:
        """Initialize TileNet.

        Args:
            in_channels: Number of input channels.
            z_dim: Output embedding dimension.
        """
        super().__init__()
        self.in_planes: int = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.layer5 = self._make_layer(z_dim, 2, stride=2)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a sequential residual layer.

        Args:
            planes: Number of output channels for each block.
            num_blocks: Number of residual blocks in the layer.
            stride: Stride of the first block.

        Returns:
            A sequential container of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, s in enumerate(strides):
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute TileNet embeddings.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        return x.view(x.size(0), -1)


def tilenet(
    weights: TileNet_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """TileNet (Tile2Vec) encoder.

    .. versionadded:: 0.9

    Args:
        weights: Pre-trained TileNet weights to load.
        *args: Positional arguments.
        **kwargs: Keyword arguments forwarded to model.

    Returns:
        A TileNet model.
    """
    if weights:
        kwargs['in_channels'] = weights.meta['in_chans']

    model = TileNet(*args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=True
        )
        assert missing_keys == []
        assert unexpected_keys == []

    return model
