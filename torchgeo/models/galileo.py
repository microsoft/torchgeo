# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/nasaharvest/galileo. Copyright (c) nasaharvest

"""Galileo encoder models.

Reference:
Gabriel et al., Galileo: Learning Global & Local Features of Many Remote Sensing Modalities

"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum
from torchvision.transforms import Resize

GalileoVariant = Literal['nano', 'tiny', 'base']

_GALILEO_CONFIGS = {
    'nano': {'embed_dim': 192, 'depth': 6, 'num_heads': 3},
    'tiny': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
}

IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 4
MLP_RATIO = 4.0


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GalileoEncoder(nn.Module):
    """Encoder-only Galileo vision transformer.

    Supported variants
    - nano
    - tiny
    - base
    """

    def __init__(self, variant: GalileoVariant = 'base') -> None:
        """Initialize a Galileo encoder.

        Args:
            variant: Model size variant ('nano', 'tiny', or 'base').
        """
        super().__init__()

        if variant not in _GALILEO_CONFIGS:
            raise ValueError(f'Unknown Galileo variant: {variant}')

        cfg = _GALILEO_CONFIGS[variant]
        embed_dim = cfg['embed_dim']
        depth = cfg['depth']
        num_heads = cfg['num_heads']

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            IN_CHANNELS, embed_dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )

        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

        # Fixed-size positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                _Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=MLP_RATIO)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image embeddings.

        Args:
            x: Input tensor of shape (B, 4, 224, 224).

        Returns:
            A tensor of shape (B, D) with image embeddings.
        """
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f'Expected input size {(IMAGE_SIZE, IMAGE_SIZE)}, '
                f'got {tuple(x.shape[-2:])}'
            )

        # Patchify
        x = self.patch_embed(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        # Normalize + mean pool
        x = self.norm(x)
        x = x.mean(dim=1)

        return x


class GalileoWeights(WeightsEnum):
    """Pretrained weights for Galileo encoders."""

    GALILEO_S2_NANO_V1 = Weights(
        url='https://huggingface.co/nasaharvest/galileo/resolve/0bbc53293a37dea7d563566f015c5527bdaf3793/models/nano/encoder.pt',
        transforms=partial(Resize, size=(IMAGE_SIZE, IMAGE_SIZE)),
        meta={
            'variant': 'nano',
            'in_channels': IN_CHANNELS,
            'embed_dim': 192,
            'dataset': 'Sentinel-2',
            'license': 'MIT',
        },
    )

    GALILEO_S2_TINY_V1 = Weights(
        url='https://huggingface.co/nasaharvest/galileo/resolve/87d646dd7b63f14d9610694d87f8dc7b5912a6df/models/tiny/encoder.pt',
        transforms=partial(Resize, size=(IMAGE_SIZE, IMAGE_SIZE)),
        meta={
            'variant': 'tiny',
            'in_channels': IN_CHANNELS,
            'embed_dim': 384,
            'dataset': 'Sentinel-2',
            'license': 'MIT',
        },
    )

    GALILEO_S2_BASE_V1 = Weights(
        url='https://huggingface.co/nasaharvest/galileo/resolve/f039dd5dde966a931baeda47eb680fa89b253e4e/models/base/encoder.pt',
        transforms=partial(Resize, size=(IMAGE_SIZE, IMAGE_SIZE)),
        meta={
            'variant': 'base',
            'in_channels': IN_CHANNELS,
            'embed_dim': 768,
            'dataset': 'Sentinel-2',
            'license': 'MIT',
        },
    )


def galileo(*, weights: GalileoWeights | None = None, **kwargs: Any) -> GalileoEncoder:
    """Galileo encoder factory."""
    model = GalileoEncoder()

    if weights is not None:
        weights = GalileoWeights.verify(weights)
        state_dict = weights.get_state_dict(progress=True)
        model.load_state_dict(state_dict, strict=False)

    return model
