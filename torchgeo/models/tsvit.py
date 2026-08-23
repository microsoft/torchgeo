# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Time-Series Vision Transformer (TSViT) model."""

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PreNorm(nn.Module):
    """Pre-normalization layer for Transformer."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Initialize the PreNorm layer."""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass."""
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward layer for Transformer."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """Initialize the FeedForward layer."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.net(x)


class Attention(nn.Module):
    """Standard multi-head self-attention layer."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        """Initialize the Attention layer."""
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        _, _, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the block."""
        super().__init__()
        self.attn = PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        )
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Module):
    """Standard Transformer encoder."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the Transformer."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TSViT(nn.Module):
    """Time-Series Vision Transformer (TSViT).

    This model processes Satellite Image Time Series (SITS).

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2301.11942
    """

    def __init__(
        self,
        img_res: int = 24,
        patch_size: int = 3,
        num_channels: int = 14,
        num_classes: int = 20,
        max_seq_len: int = 16,
        dim: int = 128,
        temporal_depth: int = 10,
        spatial_depth: int = 4,
        heads: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        pool: str = "cls",
        scale_dim: int = 4,
    ) -> None:
        """Initialize the TSViT model."""
        super().__init__()
        self.image_size = img_res
        self.patch_size = patch_size
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = num_classes
        self.num_frames = max_seq_len
        self.dim = dim
        self.temporal_depth = temporal_depth
        self.spatial_depth = spatial_depth
        self.heads = heads
        self.dim_head = dim_head
        self.pool = pool
        self.scale_dim = scale_dim

        assert self.pool in {"cls", "mean"}, "pool type must be either cls or mean"
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by patch size."

        num_patches = self.num_patches_1d**2
        patch_dim = (num_channels - 1) * self.patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )

        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(
            self.dim,
            self.temporal_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            dropout,
        )

        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(
            self.dim,
            self.spatial_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            dropout,
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.patch_size**2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        x = x.permute(0, 1, 4, 2, 3)
        B, T = x.shape[:2]

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]

        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)

        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(
            B, T, self.dim
        )

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)

        cls_temporal_tokens = repeat(
            self.temporal_token, "() N d -> b N d", b=B * self.num_patches_1d**2
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)

        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2)

        x = rearrange(
            x,
            "b c (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)",
            h1=self.num_patches_1d,
            w1=self.num_patches_1d,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x