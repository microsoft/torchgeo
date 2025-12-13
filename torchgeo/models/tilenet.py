# Copyright (c) TorchGeo Contributors.
# Licensed under the MIT License.

"""TileNet model from Tile2Vec.

This module implements the CIFAR-style ResNet encoder used in Tile2Vec.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["TileNet"]


class BasicBlock(nn.Module):
    """Standard CIFAR-style ResNet BasicBlock."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        """Initialize a BasicBlock.

        Args:
            in_planes: Number of input feature channels.
            planes: Number of output feature channels.
            stride: Convolution stride for the first layer.
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

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass of the residual block."""
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class TileNet(nn.Module):
    """TileNet encoder used in Tile2Vec.

    This is a CIFAR-style ResNet with:
    - a 3*3 stride-1 stem (no maxpool),
    - 5 residual stages,
    - and no classification head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        z_dim: int = 512,
        num_blocks: list[int] | None = None,
    ) -> None:
        """Initialize a TileNet model.

        Args:
            in_channels: Number of input channels.
            z_dim: Output embedding dimension.
            num_blocks: Number of residual blocks in each stage.
                Must be a list of length 5. Defaults to [2, 2, 2, 2, 2].
        """
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2, 2]

        if len(num_blocks) != 5:
            raise ValueError("num_blocks must have length 5.")

        self.in_planes = 64

        # CIFAR-style stem
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Residual stages
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(z_dim, num_blocks[4], stride=2)

        self.out_dim = z_dim

        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a residual stage composed of several BasicBlocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the TileNet encoder.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, z_dim) containing tile embeddings.
        """
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)

        return x
