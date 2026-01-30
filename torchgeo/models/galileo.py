# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/nasaharvest/galileo. Copyright (c) Presto Authors.

"""Galileo encoder models.

Reference:
Gabriel et al., Galileo: Learning Global & Local Features of Many Remote Sensing Modalities
"""
import collections.abc
import itertools
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention  # type: ignore[attr-defined]
from torch import Tensor, vmap
from torchvision.models._api import Weights, WeightsEnum
from torchvision.transforms import Resize

BASE_GSD = 10

_GALILEO_CONFIGS = {
    'nano': {'embed_dim': 192, 'depth': 6, 'num_heads': 3},
    'tiny': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
}

IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 4
MLP_RATIO = 4.0
# band information
S1_BANDS = ["VV", "VH"]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
TC_BANDS = ["def", "soil", "aet"]
VIIRS_BANDS = ["avg_rad"]
SRTM_BANDS = ["elevation", "slope"]
DW_BANDS = [
    "DW_water",
    "DW_trees",
    "DW_grass",
    "DW_flooded_vegetation",
    "DW_crops",
    "DW_shrub_and_scrub",
    "DW_built",
    "DW_bare",
    "DW_snow_and_ice",
]
WC_BANDS = [
    "WC_temporarycrops",
    "WC_maize",
    "WC_wintercereals",
    "WC_springcereals",
    "WC_irrigation",
]
STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]

LANDSCAN_BANDS = ["b1"]
LOCATION_BANDS = ["x", "y", "z"]

SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS


SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int,
    grid_size: int,
    res: Tensor,
    cls_token: bool = False,
    device: str = "cpu",
) -> Tensor:
    """Generate 2D sine-cosine positional embeddings with resolution scaling.

    Args:
        embed_dim (int): Dimension of the positional embedding.
        grid_size (int): Height and width of the spatial grid.
        res (Tensor): Resolutions for each sample, shape (N,).
        cls_token (bool, optional): Whether to prepend a class token. Defaults to False.
        device (str, optional): Device for tensors. Defaults to "cpu".

    Returns:
        Tensor: Positional embeddings of shape
            (N, grid_size*grid_size, embed_dim)
            or
            (N, 1+grid_size*grid_size, embed_dim) if cls_token=True.
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    gw, gh = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack([gw, gh], dim=0)

    grid = torch.einsum("chw,n->cnhw", grid, res)
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor,) -> Tensor:
    """Generate 2D sine-cosine embeddings from a spatial grid.

    Args:
        embed_dim (int): Embedding dimensionality (must be even).
        grid (Tensor): A tensor of shape (2, N, H, W) representing XY coordinates.

    Returns:
        Tensor: Flattened positional embedding of shape (N*H*W, embed_dim).
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor,) -> Tensor:
    """Generate 1D sine-cosine positional embeddings.

    Args:
        embed_dim (int): Embedding dimensionality (must be even).
        pos (Tensor): Positions to encode, shape (M,).

    Returns:
        Tensor: Sine-cosine embeddings of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def get_month_encoding_table(embed_dim: int) -> Tensor:
    """Create sinusoidal month encoding table for 12 months.

    Args:
        embed_dim (int): Embedding dimensionality (must be even).

    Returns:
        Tensor: Month embedding table of shape (12, embed_dim).
    """
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], dim=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], dim=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], dim=-1)

    return month_table


def to_2tuple(x: Any) -> tuple[Any, Any]:
    """Convert input into a 2-tuple.

    Args:
        x (Any): Input object.

    Returns:
        tuple: A tuple of length 2.
    """
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))



# thanks to https://github.com/bwconrad/flexivit/ for this nice implementation
# of the FlexiPatchEmbed module
class FlexiPatchEmbed(nn.Module):
    """Flexible 2D patch embedding layer supporting multiple patch sizes.

    This module computes patch embeddings using a base convolution kernel,
    and can dynamically resize the kernel to different patch sizes using
    pseudo-inverse resampling (FlexiViT approach).

    Args:
        patch_size (int | tuple[int, int]): Base patch size used for the learnable kernel.
        in_chans (int): Number of input image channels.
        embed_dim (int): Dimension of output embeddings.
        bias (bool): Whether to include bias in the convolution.
        patch_size_seq (Sequence[int]): List of allowed patch sizes for random sampling.
        interpolation (str): Interpolation mode used when resizing patch kernels.
        antialias (bool): Whether to apply anti-aliasing when resizing kernels.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
        embed_dim: int = 128,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Initialize FlexiPatchEmbed."""
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

        self.interpolation = interpolation
        self.antialias = antialias
        self.patch_size_seq = patch_size_seq

        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict[tuple[int, int], Tensor]:
        """Precompute pseudo-inverse matrices for different patch sizes.

        Returns:
            dict: Mapping from patch size tuple to pseudo-inverse resize matrix.
        """
        pinvs = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """Resize input tensor using interpolation.

        Args:
            x (Tensor): Input tensor of shape (H, W).
            shape (tuple[int, int]): Target (height, width).

        Returns:
            Tensor: Resized tensor of shape (shape).
        """
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> Tensor:
        """Compute pseudo-inverse resize matrix between patch sizes.

        Args:
            old_shape (tuple[int, int]): Original patch kernel shape.
            new_shape (tuple[int, int]): Target patch kernel shape.

        Returns:
            Tensor: Pseudo-inverse transform matrix.
        """
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return cast(Tensor, torch.linalg.pinv(resize_matrix))
    
    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: tuple[int, int]) -> Tensor:
        """Resize patch embedding weights to a new patch size.

        Args:
            patch_embed (Tensor): Original convolution kernel.
            new_patch_size (tuple[int, int]): Target patch size.

        Returns:
            Tensor: Resized patch embedding kernel.
        """
        if self.patch_size == new_patch_size:
            return patch_embed

        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)

        pinv = self.pinvs[new_patch_size].to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor)-> Tensor:
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
        return cast(Tensor, v_resample_patch_embed(patch_embed))

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """Forward pass to produce patch embeddings.

        Args:
            x (Tensor): Input tensor of shape [B, H, W, (T), C].
            patch_size (int | tuple[int, int] | None): Patch size override.

        Returns:
            Tensor: Patch-embedded output with spatial dimensions reduced.
        """
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0
        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            patch_size = self.patch_size

        patch_size = to_2tuple(patch_size)

        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=batch_size, t=num_timesteps)
        else:
            x = rearrange(x, "b c h w -> b h w c")

        return x

class Mlp(nn.Module):
    """Two-layer MLP block used in Vision Transformers.

    Args:
        in_features (int): Input dimension.
        hidden_features (int | None): Hidden layer dimension.
        out_features (int | None): Output dimension.
        act_layer (nn.Module): Activation layer class.
        bias (bool): Whether to use bias in linear layers.
        drop (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """Initialize MLP block."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Apply MLP transformation.

        Args:
            x (Tensor): Input tensor of shape (B, N, C).

        Returns:
            Tensor: Output tensor of same shape.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    """Simple layer scaling module.

    Args:
        dim (int): Feature dimension.
        init_values (float): Initial scaling factor.
        inplace (bool): Whether to use in-place multiplication.
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale inputs by learnable factor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Scaled tensor.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False,)-> Tensor:
    """Stochastic depth: randomly drop residual paths.

    Args:
        x (Tensor): Input tensor.
        drop_prob (float): Drop probability.
        training (bool): Whether the model is in training mode.

    Returns:
        Tensor: Output tensor with stochastic dropping applied.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Layer module wrapping stochastic depth.

    Args:
        drop_prob (float | None): Probability of dropping paths.
    """
    drop_prob: float | None

    def __init__(self, drop_prob: float | None = None) -> None:
        """Initialize DropPath module."""
        super().__init__()
        self.drop_prob = drop_prob


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass applying stochastic depth.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output with stochastic depth applied.
        """
        return drop_path(x,self.drop_prob if self.drop_prob is not None else 0.0, self.training)


class Block(nn.Module):
    """Transformer block with attention, MLP, normalization, and optional cross-attention.

    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden size to input size.
        qkv_bias (bool): Whether to include bias in qkv projections.
        drop (float): Dropout probability.
        attn_drop (float): Attention dropout rate.
        drop_path (float): Stochastic depth drop probability.
        init_values (float | None): Initial gamma scaling value.
        act_layer (nn.Module): Activation class.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward transformer block.

        Args:
            x (Tensor): Input tensor of shape (B, N, C).
            y (Tensor | None): Optional cross-attention source.
            attn_mask (Tensor | None): Boolean attention mask.

        Returns:
            Tensor: Output tensor of shape (B, N, C).
        """
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ModuleListWithInit(nn.ModuleList):
    """ModuleList with custom weight initialization.

    This subclass of nn.ModuleList adds an `_init_weights` method that can be
    applied to child modules to initialize linear layers with Xavier uniform
    initialization.
    """

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights of linear layers.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GalileoBase(nn.Module):
    """Base class containing shared logic for Galileo encoders.

    This class manages:
      - Band group definitions (space-time, time, static, etc.)
      - Positional, month, and channel embeddings
      - Collapsing and expanding mixed spatiotemporal token structures
      - Mask handling (token deletion, reinsertion)
      - Token encoding combining spatial, temporal, and channel embeddings

    Args:
        embedding_size (int): Size of the embedding dimension.
        depth (int): Number of Transformer blocks.
        mlp_ratio (float): Expansion ratio for MLP layers.
        num_heads (int): Number of attention heads.
        max_sequence_length (int): Maximum temporal sequence length.
        base_patch_size (int): Default patch size.
        use_channel_embs (bool): Whether channel embeddings are learnable.
        drop_path (float): Stochastic depth drop probability.
    """


    def __init__(
        self,
        space_time_groups: dict[str, Any],
        space_groups: dict[str, Any],
        time_groups: dict[str, Any],
        static_groups: dict[str, Any],

        *,
        embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        base_patch_size: int = 4,
        use_channel_embs: bool = True,
        drop_path: float = 0.0,
        **kwargs: Any,
    )-> None:
        """Initialize the Galileo encoder."""
        super().__init__()

        self.space_time_groups = space_time_groups
        self.space_groups = space_groups
        self.time_groups = time_groups
        self.static_groups = static_groups

        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size
        

        self.blocks = ModuleListWithInit(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

        self.max_sequence_length = max_sequence_length

        # Positional embeddings (time-only)
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )

        # Month embeddings (non-trainable)
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        embed = nn.Embedding.from_pretrained( # type: ignore[no-untyped-call] 
            month_tab, 
            freeze=True)
        
        self.month_embed = cast(nn.Embedding, embed)

        # Channel embeddings (optionally learnable)
        args = {"requires_grad": True} if use_channel_embs else {"requires_grad": False}

        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(self.space_time_groups), int(embedding_size * 0.25)), **args
        )
        self.sp_channel_embed = nn.Parameter(
            torch.zeros(len(self.space_groups), int(embedding_size * 0.25)), **args
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(self.time_groups), int(embedding_size * 0.25)), **args
        )
        self.st_channel_embed = nn.Parameter(
            torch.zeros(len(self.static_groups), int(embedding_size * 0.25)), **args
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights.

        Args:
            m (nn.Module):  Module whose weights need to be initialized.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def collapse_and_combine_hwtc(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    )-> tuple[Tensor, Tensor]:
        """Flatten and concatenate heterogeneous spatiotemporal token groups.

        Args:
            s_t_x (Tensor): Space-time tokens, shape (B, H, W, T, Cg, D).
            sp_x (Tensor): Space-only tokens, shape (B, H, W, Cg, D).
            t_x (Tensor): Time-only tokens, shape (B, T, Cg, D).
            st_x (Tensor): Static tokens, shape (B, Cg, D).
            s_t_m (Tensor): Mask for space-time tokens.
            sp_m (Tensor): Mask for spatial-only tokens.
            st_m (Tensor): Mask for static tokens.
            t_m (Tensor): Mask for time tokens.


        Returns:
            tuple[Tensor, Tensor]: Flattened tokens and masks of shape (B, N, D) and (B, N).
        """
        s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
        sp_x = rearrange(sp_x, "b h w c_g d -> b (h w c_g) d")
        t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")

        s_t_m = rearrange(s_t_m, "b h w t c_g -> b (h w t c_g)")
        sp_m = rearrange(sp_m, "b h w c_g -> b (h w c_g)")
        t_m = rearrange(t_m, "b t c_g -> b (t c_g)")

        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=1)
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=1)
        return x, m

    @classmethod
    def split_and_expand_hwtc(
        cls,
        x: torch.Tensor,
        h: int,
        w: int,
        t: int,
        s_t_c_g: int,
        sp_c_g: int,
        st_c_g: int,
        t_c_g: int,
        
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Undo the flattening from collapse_and_combine_hwtc.

        Args:
            x (Tensor): Flattened features, shape (B, N, D).
            h (int): Height.
            w (int): Width.
            t (int): Time steps.
            s_t_c_g (int): Number of space-time channel groups.
            sp_c_g (int): Number of spatial-only channel groups.
            st_c_g (int): Number of static channel groups.
            t_c_g (int): Number of time channel groups.


        Returns:
            tuple[Tensor]: Restored structured token groups.
        """
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g

        s_t_x = rearrange(x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g)
        sp_x = rearrange(
            x[:, n_s_t_t : -(n_t_t + st_c_g)],
            "b (h w c) d -> b h w c d",
            h=h,
            w=w,
            c=sp_c_g,
        )
        t_x = rearrange(x[:, -(n_t_t + st_c_g) : -st_c_g], "b (t c) d -> b t c d", t=t, c=t_c_g)
        st_x = x[:, -st_c_g:]

        return s_t_x, sp_x, t_x, st_x

    def apply_encodings(
    self,
    s_t_x: Tensor,
    sp_x: Tensor,
    t_x: Tensor,
    st_x: Tensor,
    months: Tensor,
    patch_size: int,
    input_res: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply positional, month, spatial, and channel encodings to tokens.

        Args:
            s_t_x (Tensor): Spatiotemporal token features.
            sp_x (Tensor): Spatial token features.
            t_x (Tensor): Temporal token features.
            st_x (Tensor): Static token features.
            months (Tensor): Month indices for each timestep.
            patch_size (int): Patch size used in patch embedding.
            input_res (float): Ground sample distance of input.

        Returns:
            tuple[Tensor]: Encoded versions of each token group.
        """
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g = sp_x.shape[-2], t_x.shape[-2]
        st_c_g = st_x.shape[-2]

        s_t_channel = repeat(self.s_t_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t)
        t_channel = repeat(self.t_channel_embed, "c_g d -> b t c_g d", b=b, t=t)
        st_channel = repeat(self.st_channel_embed, "c_g d -> b c_g d", b=b)
        sp_channel = repeat(self.sp_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w)

        pos_embed_s_t = repeat(self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=s_t_c_g)
        m_embed_s_t = repeat(self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=s_t_c_g)

        pos_embed_t = repeat(self.pos_embed[:t], "t d -> b t c_g d", b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), "b t d -> b t c_g d", c_g=t_c_g)

        t_zeros = torch.zeros(b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device)
        sp_zeros = torch.zeros(b, h, w, sp_c_g, sp_channel.shape[-1] * 2, device=sp_channel.device)
        st_zeros = torch.zeros(b, st_c_g, st_channel.shape[-1] * 3, device=st_channel.device)

        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert h == w, "get_2d_sincos_pos_embed_with_resolution requires h==w"
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25),
            h,
            torch.ones(b).to(s_t_x.device) * gsd_ratio,
            device=str(s_t_x.device),
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)

        spatial_embed_s_t = repeat(
            spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=s_t_c_g
        )
        spatial_embed_s = repeat(
            spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=sp_c_g
        )

        s_t_embed = torch.cat([s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1)
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)

        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed


class Encoder(GalileoBase):
    """Galileo encoder implementing flexible patch embeddings and Transformer layers.

    Extends `GalileoBase` by adding:
      - Group-specific projection modules (space-time, space, time, static)
      - Linear projections for non-image inputs
      - Attention application with masking and optional early-exit tokens
      - Output normalization and final structuring of token groups

    Args:
        max_patch_size (int): Maximum patch size for flexible embedding.
        embedding_size (int): Embedding dimension.
        depth (int): Number of Transformer blocks.
        mlp_ratio (float): MLP hidden expansion ratio.
        num_heads (int): Number of attention heads.
        max_sequence_length (int): Maximum temporal sequence length.
        freeze_projections (bool): Whether projection layers should be frozen.
        drop_path (float): Stochastic depth drop probability.
    """

    def __init__(

        self,
        space_time_groups: dict[str, Any],
        space_groups: dict[str, Any],
        time_groups: dict[str, Any],
        static_groups: dict[str, Any],

        *,
        max_patch_size: int = 8,
        embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        freeze_projections: bool = False,
        drop_path: float = 0.0,
    )-> None:
        """Initialize the Galileo encoder."""
        super().__init__(
            space_time_groups=space_time_groups,
            space_groups=space_groups,
            time_groups=time_groups,
            static_groups=static_groups,
            embedding_size=embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            base_patch_size=max_patch_size,
            use_channel_embs=True,
            drop_path=drop_path,
        )
        self.space_time_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group),
                    embed_dim=embedding_size,
                    patch_size=max_patch_size,
                )
                for group_name, group in self.space_time_groups.items()
            }
        )

        self.space_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group),
                    embed_dim=embedding_size,
                    patch_size=max_patch_size,
                )
                for group_name, group in self.space_groups.items()
            }
        )

        self.time_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(
                    in_features=len(group),
                    out_features=embedding_size,
                )
                for group_name, group in self.time_groups.items()
            }
        )

        self.static_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(
                    in_features=len(group),
                    out_features=embedding_size,
                )
                for group_name, group in self.static_groups.items()
            }
        )

        if freeze_projections:
            self.space_time_embed.requires_grad_(False)
            self.space_embed.requires_grad_(False)
            self.time_embed.requires_grad_(False)
            self.static_embed.requires_grad_(False)

        self.norm = nn.LayerNorm(embedding_size)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights using Xavier uniform for linear layers.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def apply_linear_projection(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
        patch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply group-specific linear or convolutional projections.

        Outputs shape:
            s_t_x  → (B, H', W', T, Cg, D)
            sp_x   → (B, H', W', Cg, D)
            t_x    → (B, T, Cg, D)
            st_x   → (B, Cg, D)

        Args:
            s_t_x (Tensor): Space-time features [B,H,W,T,C].
            sp_x (Tensor): Space-only features [B,H,W,C].
            t_x (Tensor): Time-only features [B,T,C].
            st_x (Tensor): Static features [B,C].
            s_t_m (Tensor): Mask for space-time tokens.
            sp_m (Tensor): Mask for spatial-only tokens.
            st_m (Tensor): Mask for static tokens.
            t_m (Tensor): Mask for time tokens.
            patch_size (int): Patch size for spatial projections.

        Returns:
            tuple[Tensor]: Projected features & masks for all groups.
        """
        b, h, w, t, _ = s_t_x.shape
        new_h, new_w = h // patch_size, w // patch_size

        s_t_l, sp_l, t_l, st_l = [], [], [], []
        s_t_m_l, sp_m_l, t_m_l, st_m_l = [], [], [], []

        # space-time projections
        for idx, (group_name, channel_idxs) in enumerate(self.space_time_groups.items()):
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            if s_t_m_l[-1].min() == 0:
                s_t_l.append(
                    self.space_time_embed[group_name](
                        s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                s_t_l.append(
                    torch.zeros(
                        b, new_h, new_w, t, self.embedding_size,
                        dtype=s_t_x.dtype, device=s_t_x.device
                    )
                )

        # space-only projections
        for idx, (group_name, channel_idxs) in enumerate(self.space_groups.items()):
            sp_m_l.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            if sp_m_l[-1].min() == 0:
                sp_l.append(
                    self.space_embed[group_name](
                        sp_x[:, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                sp_l.append(
                    torch.zeros(
                        b, new_h, new_w, self.embedding_size,
                        dtype=sp_x.dtype, device=sp_x.device
                    )
                )

        # time-only projections
        for idx, (group_name, channel_idxs) in enumerate(self.time_groups.items()):
            t_m_l.append(t_m[:, :, idx])
            if t_m_l[-1].min() == 0:
                t_l.append(self.time_embed[group_name](t_x[:, :, channel_idxs]))
            else:
                t_l.append(
                    torch.zeros(b, t, self.embedding_size, dtype=t_x.dtype, device=t_x.device)
                )

        # static projections
        for idx, (group_name, channel_idxs) in enumerate(self.static_groups.items()):
            st_m_l.append(st_m[:, idx])
            if st_m_l[-1].min() == 0:
                st_l.append(self.static_embed[group_name](st_x[:, channel_idxs]))
            else:
                st_l.append(
                    torch.zeros(b, self.embedding_size, dtype=st_x.dtype, device=st_x.device)
                )

        return (
            torch.stack(s_t_l, dim=-2),
            torch.stack(sp_l, dim=-2),
            torch.stack(t_l, dim=-2),
            torch.stack(st_l, dim=-2),
            torch.stack(s_t_m_l, dim=-1),
            torch.stack(sp_m_l, dim=-1),
            torch.stack(t_m_l, dim=-1),
            torch.stack(st_m_l, dim=-1),
        )

    @staticmethod
    def remove_masked_tokens(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Remove masked tokens from sequences for accelerated attention.

        Args:
            x (Tensor): Input tokens [B, N, D].
            mask (Tensor): Mask [B, N], where 1=masked, 0=keep.

        Returns:
            tuple:
                - Tensor: Unmasked tokens.
                - Tensor: Indices used to restore original order later.
                - Tensor: Updated mask for trimmed sequence.
        """
        org_mask_dtype = mask.dtype
        mask = mask.bool()

        sorted_mask, indices = torch.sort(
            (~mask).int(), dim=1, descending=True, stable=True
        )
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)

        max_len = sorted_mask.sum(-1).max()
        x = x[:, :max_len]
        new_mask = 1 - sorted_mask[:, :max_len]

        return x, indices, new_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def add_removed_tokens(
    x: Tensor,
    indices: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
        """Restore masked tokens after attention into their original positions.

        Args:
            x (Tensor): Token sequence after attention.
            indices (Tensor): Original positions from remove_masked_tokens.
            mask (Tensor): Mask of padded/masked positions.

        Returns:
            tuple:
                - Tensor: Restored sequence with masked tokens zero-filled.
                - Tensor: Restored mask.
        """
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.ones(
                    (x.shape[0], indices.shape[1] - x.shape[1]),
                    device=x.device,
                    dtype=mask.dtype,
                ),
            ),
            dim=-1,
        )

        out = masked_tokens.clone()
        out[~full_mask.bool()] = x[~mask.bool()]
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)

        return out, full_mask

    def apply_attn(
        self,
        s_t_x: Tensor,
        sp_x: Tensor,
        t_x: Tensor,
        st_x: Tensor,
        s_t_m: Tensor,
        sp_m: Tensor,
        t_m: Tensor,
        st_m: Tensor,
        months: Tensor,
        patch_size: int,
        input_res: float,
        exit_after: int | None,
        token_exit_cfg: dict[str, Any] | None,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor
    ]:
        """Apply Transformer blocks with masking, positional encodings and optional token-exit logic.

        Args:
            s_t_x (Tensor): Space-time tokens.
            sp_x (Tensor): Spatial-only tokens.
            t_x (Tensor): Time tokens.
            st_x (Tensor): Static tokens.

            s_t_m (Tensor): Mask for space-time tokens.
            sp_m (Tensor): Mask for spatial-only tokens.
            t_m (Tensor): Mask for time tokens.
            st_m (Tensor): Mask for static tokens.

            months (Tensor): Month indices.
            patch_size (int): Patch size used.
            input_res (float): Input resolution in meters.
            exit_after (int | None): Layer after which to exit.
            token_exit_cfg (dict | None): Mapping of token groups to 
            early exit layers.

        Returns:
            tuple[Tensor]: Updated tokens and masks after attention.
        """
        if token_exit_cfg:
            exit_s_t, exit_sp, exit_t, exit_st = self.create_token_exit_ids(
                s_t_x, sp_x, t_x, st_x, token_exit_cfg
            )
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(
                exit_s_t, exit_sp, exit_t, exit_st, s_t_m, sp_m, t_m, st_m
            )
            exited_tokens, _ = self.collapse_and_combine_hwtc(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
            )
        else:
            exit_ids_seq = None
            exited_tokens = None

        _, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g, st_c_g = sp_x.shape[3], t_x.shape[-2], st_x.shape[-2]

        s_t_x, sp_x, t_x, st_x = self.apply_encodings(
            s_t_x, sp_x, t_x, st_x, months, patch_size, input_res
        )

        x, m = self.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        new_m = m >= 1
        x, indices, new_m = self.remove_masked_tokens(x, new_m)

        if exit_ids_seq is not None:
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, m >= 1)
            assert exited_tokens is not None
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, m >= 1)
            assert exited_tokens is not None
            final_exited_tokens = exited_tokens
            

        for i_blk, blk in enumerate(self.blocks):
            if (exit_after is not None) and ((i_blk + 1) > exit_after):
                break

            if (exit_ids_seq is not None) and (i_blk > 0):
                exited_tokens = torch.where(
                    exit_ids_seq == i_blk,
                    x.detach(),
                    final_exited_tokens.detach(),
                )

            x = blk(x)


        if exit_ids_seq is not None:
            x = torch.where(
                exit_ids_seq == (i_blk + 1),
                x.detach(),
                final_exited_tokens.detach(),
            )

        x, _ = self.add_removed_tokens(x, indices, new_m)

        return (
            *self.split_and_expand_hwtc(x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g),
            s_t_m,
            sp_m,
            t_m,
            st_m,
        )

    @classmethod
    def average_tokens(
        cls,
        s_t_x: Tensor,
        sp_x: Tensor,
        t_x: Tensor,
        st_x: Tensor,
        s_t_m: Tensor,
        sp_m: Tensor,
        t_m: Tensor,
        st_m: Tensor,
    ) -> Tensor:
        """Average unmasked tokens across all token groups.

        Args:
        s_t_x (Tensor): Space-time tokens.
        sp_x (Tensor): Spatial-only tokens.
        t_x (Tensor): Time tokens.
        st_x (Tensor): Static tokens.

        s_t_m (Tensor): Mask for space-time tokens.
        sp_m (Tensor): Mask for spatial-only tokens.
        t_m (Tensor): Mask for time tokens.
        st_m (Tensor): Mask for static tokens.


        Returns:
            Tensor: Mean embedding per sample.
        """
        x, m = cls.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        x, _, m = cls.remove_masked_tokens(x, m)
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    @classmethod
    def apply_mask_and_average_tokens_per_patch(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ) -> Tensor:
        """Average tokens per spatial tile, handling temporal and static tokens.

        Args:
        s_t_x (Tensor): Space-time tokens.
        sp_x (Tensor): Spatial-only tokens.
        t_x (Tensor): Time tokens.
        st_x (Tensor): Static tokens.

        s_t_m (Tensor): Mask for space-time tokens.
        sp_m (Tensor): Mask for spatial-only tokens.
        t_m (Tensor): Mask for time tokens.
        st_m (Tensor): Mask for static tokens.


        Returns:
            Tensor: Averaged tile-level embeddings.
        """
        s_t_x = rearrange(s_t_x, "b t_h t_w t c_g d -> b (t_h t_w) (t c_g) d")
        sp_x = rearrange(sp_x, "b t_h t_w c_g d -> b (t_h t_w) c_g d")

        t_x = repeat(
            rearrange(t_x, "b t c_g d -> b (t c_g) d"),
            "b n d -> b s n d",
            s=sp_x.shape[1],
        )
        st_x = repeat(st_x, "b c_g d -> b s c_g d", s=sp_x.shape[1])

        s_t_m = rearrange(s_t_m, "b t_h t_w t c_g -> b (t_h t_w) (t c_g)")
        sp_m = rearrange(sp_m, "b t_h t_w c_g -> b (t_h t_w) c_g")
        t_m = repeat(
            rearrange(t_m, "b t c_g -> b (t c_g)"),
            "b n -> b s n",
            s=sp_x.shape[1],
        )
        st_m = repeat(st_m, "b c_g -> b s c_g", s=sp_x.shape[1])

        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=2)
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=2)

        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=2) / torch.sum(1 - m, -1, keepdim=True)

    def create_token_exit_ids(
        self,
        s_t_x: Tensor,
        sp_x: Tensor,
        t_x: Tensor,
        st_x: Tensor,
        token_exit_cfg: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Create early-exit IDs for token groups.

        Args:
            s_t_x (Tensor): Space-time tokens.
            sp_x (Tensor): Spatial-only tokens.
            t_x (Tensor): Time tokens.
            st_x (Tensor): Static tokens.

            token_exit_cfg (dict): Mapping group names → exit depth index.

        Returns:
            tuple[Tensor]: Exit ID tensors aligned with token shapes.
        """
        exit_s_t = torch.zeros_like(s_t_x)
        exit_sp = torch.zeros_like(sp_x)
        exit_t = torch.zeros_like(t_x)
        exit_st = torch.zeros_like(st_x)

        for idx, (key, _) in enumerate(self.space_time_groups.items()):
            exit_s_t[:, :, :, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.space_groups.items()):
            exit_sp[:, :, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.time_groups.items()):
            exit_t[:, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.static_groups.items()):
            exit_st[:, idx, :] = token_exit_cfg[key]

        return exit_s_t, exit_sp, exit_t, exit_st

    def forward(
        self,
        s_t_x: Tensor,
        sp_x: Tensor,
        t_x: Tensor,
        st_x: Tensor,
        s_t_m: Tensor,
        sp_m: Tensor,
        t_m: Tensor,
        st_m: Tensor,
        months: Tensor,
        patch_size: int,
        input_resolution_m: int | None = BASE_GSD,
        exit_after: int | None = None,
        token_exit_cfg: dict[str, Any] | None = None,
        add_layernorm_on_exit: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of the Galileo Encoder.

        Applies:
          - Linear / conv group projections
          - Positional + month + channel encodings
          - Transformer layers
          - Optional early exit
          - Optional output normalization

        Args:
            s_t_x (Tensor): Space-time tokens.
            sp_x (Tensor): Spatial-only tokens.
            t_x (Tensor): Time tokens.
            st_x (Tensor): Static tokens.

            s_t_m (Tensor): Mask for space-time tokens.
            sp_m (Tensor): Mask for spatial-only tokens.
            t_m (Tensor): Mask for time tokens.
            st_m (Tensor): Mask for static tokens.
            months (Tensor): Month indices for each timestep.
            patch_size (int): Patch size for embedding.
            input_resolution_m (float): Resolution of input pixels in meters.
            exit_after (int | None): Layer after which to stop computation.
            token_exit_cfg (dict | None): Mapping of token groups to exit layers.
            add_layernorm_on_exit (bool): Whether to apply LN to outputs.

        Returns:
            tuple[Tensor]: Updated token groups and metadata.
        """
        (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
        ) = self.apply_linear_projection(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size
        )

        if (exit_after is None) or (exit_after > 0):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_attn(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size,
                float(input_resolution_m) if input_resolution_m is not None else 0.0,
                exit_after=exit_after,
                token_exit_cfg=token_exit_cfg,
            )

        if add_layernorm_on_exit:
            s_t_x = self.norm(s_t_x)
            sp_x = self.norm(sp_x)
            t_x = self.norm(t_x)
            st_x = self.norm(st_x)

        return (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
        )


class GalileoWeights(WeightsEnum):     # type: ignore[misc]
    """Pretrained weights for Galileo encoder variants.

    Each weight entry includes:
      - URL to pretrained checkpoint
      - Default transform function (e.g., image resize)
      - Metadata containing model configuration

    Attributes:
        GALILEO_S2_NANO_V1: Pretrained 'nano' variant.
        GALILEO_S2_TINY_V1: Pretrained 'tiny' variant.
        GALILEO_S2_BASE_V1: Pretrained 'base' variant.
    """

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


def galileo(
    *,
    variant: str | None = None,
    weights: GalileoWeights | None = None,
    space_time_groups: dict[str, Any] | None,
    space_groups: dict[str, Any] | None,
    time_groups: dict[str, Any] | None,
    static_groups: dict[str, Any] | None,
    **kwargs: Any,
) -> Encoder:
    """Build a Galileo encoder instance.

    You may specify either:
      - ``variant`` :(nano/tiny/base) or
      - ``weights`` : which auto-selects the variant.

    Args:
        variant (str | None): Name of the architecture variant.
            Must be one of: {"nano", "tiny", "base"}.
        weights (GalileoWeights | None): Optional pretrained weights enum.
        space_time_groups (dict | None): Space-time band groups.
        space_groups (dict | None): Space-only band groups.
        time_groups (dict | None): Time-series band groups.
        static_groups (dict | None): Static feature band groups.

        **kwargs: Additional constructor arguments forwarded to `Encoder`.

    Raises:
        ValueError: If unknown variant is provided.

    Returns:
        Encoder: Instantiated (and optionally pretrained) Galileo encoder.
    """
    if weights is not None:
        weights = GalileoWeights.verify(weights)
        variant = weights.meta["variant"]

    if variant is None:
        variant = "base"

    if variant not in _GALILEO_CONFIGS:
        raise ValueError(
            f"Unknown Galileo variant: {variant!r}. "
            f"Available: {list(_GALILEO_CONFIGS.keys())}"
        )

    cfg = _GALILEO_CONFIGS[variant]

    assert space_time_groups is not None
    assert space_groups is not None
    assert time_groups is not None
    assert static_groups is not None

    model = Encoder(
        embedding_size=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        space_time_groups=space_time_groups,
        space_groups=space_groups,
        time_groups=time_groups,
        static_groups=static_groups,
        **kwargs,
    )

    if weights is not None:
        state_dict = weights.get_state_dict(
            progress=True,
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model
