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


# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
class TileNet_Weights(WeightsEnum):  # type: ignore[misc]
    """TileNet (Tile2Vec) weights.

    NAIP-pretrained Tile2Vec encoder.

    .. versionadded:: 0.7
    """

    NAIP = Weights(
        url=(
            'https://hf.co/pgangapurwala/'
            'TileNet_Weights.NAIP_ALL_TILE2VEC/resolve/'
            'af12210f5c130af76579ce8ec5e7036c1551ba25/'
            'TileNet_Weights.NAIP_ALL_TILE2VEC.pth'
        ),
        transforms=T.Normalize(mean=[0], std=[255], inplace=True),
        meta={
            'dataset': 'NAIP',
            'in_chans': 4,
            'model': 'tilenet',
            'ssl_method': 'tile2vec',
            'publication': 'https://arxiv.org/abs/1805.02855',
            'repo': 'https://github.com/ermongroup/tile2vec',
            'bands': ['R', 'G', 'B', 'NIR'],
        },
    )


# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Tile2Vec residual block with extra conv3 branch."""

    expansion: int = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, no_relu: bool = False
    ) -> None:
        """Initialize a BasicBlock.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Convolution stride.
            no_relu: Disable final ReLU (used in last block).
        """
        super().__init__()
        self.no_relu = no_relu

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

        if self.no_relu:
            return self.bn3(self.conv3(out))

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# -----------------------------------------------------------------------------
# TileNet model
# -----------------------------------------------------------------------------
class TileNet(nn.Module):
    """TileNet encoder (Tile2Vec NAIP exact)."""

    name: str = 'tilenet'
    embedding_dim: int = 512

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
        self.layer5 = self._make_layer(z_dim, 2, stride=2, no_relu=True)

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int, no_relu: bool = False
    ) -> nn.Sequential:
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


# -----------------------------------------------------------------------------
# Internal factory
# -----------------------------------------------------------------------------
def make_tilenet(in_channels: int = 4, z_dim: int = 512) -> TileNet:
    """Create a TileNet encoder.

    Args:
      in_channels: Number of input channels.
      z_dim: Output embedding dimension.

    Returns:
      A TileNet model instance.
    """
    return TileNet(in_channels=in_channels, z_dim=z_dim)


# -----------------------------------------------------------------------------
# Public factory (TorchGeo API)
# -----------------------------------------------------------------------------
def tilenet(
    weights: TileNet_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """TileNet (Tile2Vec) encoder.

    Args:
        weights: Pre-trained TileNet weights to load.
        *args: Positional arguments (unused, kept for API compatibility).
        **kwargs: Keyword arguments forwarded to ``make_tilenet``.

    Returns:
        A TileNet model.
    """
    if weights:
        kwargs['in_channels'] = weights.meta['in_chans']

    model = make_tilenet(*args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=True
        )
        assert missing_keys == []
        assert unexpected_keys == []

    return model


__all__ = ['TileNet', 'TileNet_Weights', 'make_tilenet', 'tilenet']
