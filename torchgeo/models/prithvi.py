# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Prithvi-EO-2.0 vision transformer encoder.

The sinusoidal positional embedding helpers extend the existing helpers
in :mod:`torchgeo.models.scale_mae` from 2D to 3D inputs.
"""

import math
import warnings
from typing import Any

import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from timm.models.vision_transformer import Block
from torch import Tensor, nn
from torchvision.models._api import Weights, WeightsEnum

_mean = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
_std = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
_prithvi_transforms = T.Normalize(mean=_mean, std=_std)


def _get_1d_sincos_pos_embed(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute 1D sin-cos positional embeddings for a sequence of positions.

    Args:
        embed_dim: Output embedding dimension. Must be even.
        pos: Positions to encode as a 1D tensor.

    Returns:
        Positional embeddings of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = torch.einsum('m,d->md', pos.reshape(-1), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def _get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int, int],
    add_cls_token: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute 3D sin-cos positional embeddings over a (time, height, width) grid.

    The embedding dimension is split between the width, height, and time axes
    in a 6:6:4 ratio. Tokens are ordered (t, h, w).

    Args:
        embed_dim: Output embedding dimension. Must be divisible by 16.
        grid_size: Number of patches along the time, height, and width axes.
        add_cls_token: Prepend a zero embedding for the class token.
        dtype: Desired output data type.

    Returns:
        Positional embeddings of shape (1 + t * h * w, embed_dim) if
        *add_cls_token* is True, else (t * h * w, embed_dim).

    Raises:
        ValueError: If *embed_dim* is not divisible by 16.
    """
    if embed_dim % 16 != 0:
        raise ValueError(f'embed_dim must be divisible by 16, got {embed_dim}')

    t_size, h_size, w_size = grid_size
    dim_w = embed_dim // 16 * 6
    dim_h = embed_dim // 16 * 6
    dim_t = embed_dim // 16 * 4

    w_embed = _get_1d_sincos_pos_embed(dim_w, torch.arange(w_size, dtype=torch.float64))
    h_embed = _get_1d_sincos_pos_embed(dim_h, torch.arange(h_size, dtype=torch.float64))
    t_embed = _get_1d_sincos_pos_embed(dim_t, torch.arange(t_size, dtype=torch.float64))

    w_embed = w_embed.repeat(t_size * h_size, 1)
    h_embed = h_embed.repeat_interleave(w_size, dim=0).repeat(t_size, 1)
    t_embed = t_embed.repeat_interleave(h_size * w_size, dim=0)

    pos_embed = torch.cat([w_embed, h_embed, t_embed], dim=1)
    if add_cls_token:
        pos_embed = torch.cat(
            [torch.zeros(1, embed_dim, dtype=torch.float64), pos_embed]
        )
    return pos_embed.to(dtype)


def _interpolate_pos_embedding(
    pos_embed: Tensor,
    grid_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    input_size: tuple[int, int, int],
    embed_dim: int,
) -> Tensor:
    """Adapt positional embeddings to the runtime input shape.

    If the number of frames differs from the pretraining value, the sin-cos
    table is recomputed for the new temporal grid. If the spatial size
    differs, the spatial embeddings are bicubically resampled.

    Args:
        pos_embed: Positional embeddings of shape (1, 1 + t * h * w, embed_dim).
        grid_size: Pretraining patch grid (time, height, width).
        patch_size: Patch size (time, height, width).
        input_size: Runtime input shape (time, height, width) in pixels.
        embed_dim: Embedding dimension.

    Returns:
        Positional embeddings matching the runtime patch grid.
    """
    t, h, w = input_size
    t_patches = t // patch_size[0]
    h_patches = h // patch_size[1]
    w_patches = w // patch_size[2]

    if [t_patches, h_patches, w_patches] == list(grid_size):
        return pos_embed

    if t_patches != grid_size[0]:
        new_grid_size = (t_patches, *grid_size[1:])
        pos_embed = _get_3d_sincos_pos_embed(
            embed_dim, new_grid_size, add_cls_token=True, dtype=pos_embed.dtype
        ).unsqueeze(0)
        grid_size = new_grid_size

    cls_pos_embed = pos_embed[:, :1]
    patch_pos_embed = (
        pos_embed[:, 1:].reshape(*grid_size, embed_dim).permute(0, 3, 1, 2)
    )
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed, size=(h_patches, w_patches), mode='bicubic', align_corners=True
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    return torch.cat([cls_pos_embed, patch_pos_embed], dim=1)


class PatchEmbed3D(nn.Module):
    """Spatio-temporal patch embedding for (B, C, T, H, W) inputs.

    .. versionadded:: 0.11
    """

    grid_size: tuple[int, int, int]

    def __init__(
        self,
        input_size: tuple[int, int, int] = (4, 224, 224),
        patch_size: tuple[int, int, int] = (1, 16, 16),
        in_chans: int = 6,
        embed_dim: int = 1024,
    ) -> None:
        """Initialize a new PatchEmbed3D instance.

        Args:
            input_size: Input size as (time, height, width) in pixels.
            patch_size: Patch size as (time, height, width) in pixels.
            in_chans: Number of input image channels.
            embed_dim: Output embedding dimension.

        Raises:
            ValueError: If *patch_size* exceeds *input_size* along any axis.
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        t_size, h_size, w_size = input_size
        tp_size, hp_size, wp_size = patch_size
        self.grid_size = (t_size // tp_size, h_size // hp_size, w_size // wp_size)
        if min(self.grid_size) < 1:
            raise ValueError(
                f'Patch size {patch_size} is bigger than input size {input_size}.'
            )
        self.num_patches = math.prod(self.grid_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """Embed the input patches.

        Args:
            x: Input images of shape (B, C, T, H, W).

        Returns:
            Patch embeddings of shape (B, T' * H' * W', embed_dim).
        """
        _, _, t, h, w = x.shape
        if t % self.patch_size[0] or h % self.patch_size[1] or w % self.patch_size[2]:
            warnings.warn(
                f'Input size {(t, h, w)} is not divisible by patch size '
                f'{self.patch_size}; excess pixels and frames are dropped.',
                stacklevel=2,
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class TemporalEncoder(nn.Module):
    """Sin-cos encoding of acquisition dates.

    Each date is encoded from its year and day-of-year values.

    .. versionadded:: 0.11
    """

    def __init__(self, embed_dim: int, trainable_scale: bool = False) -> None:
        """Initialize a new TemporalEncoder instance.

        Args:
            embed_dim: Output embedding dimension.
            trainable_scale: Learn a global scaling factor for the embeddings.
        """
        super().__init__()
        self.year_embed_dim = embed_dim // 2
        self.day_embed_dim = embed_dim - self.year_embed_dim
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(
        self, temporal_coords: Tensor, tokens_per_frame: int | None = None
    ) -> Tensor:
        """Encode the acquisition dates.

        Args:
            temporal_coords: Year and day-of-year values of shape (B, T, 2).
            tokens_per_frame: Number of patches per frame. If given, the
                embeddings are repeated to match the token sequence length.

        Returns:
            Temporal embeddings of shape (B, T, embed_dim), or
            (B, T * tokens_per_frame, embed_dim) if *tokens_per_frame* is given.
        """
        b, t = temporal_coords.shape[:2]
        years = _get_1d_sincos_pos_embed(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()
        ).reshape(b, t, -1)
        days = _get_1d_sincos_pos_embed(
            self.day_embed_dim, temporal_coords[:, :, 1].flatten()
        ).reshape(b, t, -1)
        embedding = self.scale * torch.cat([years, days], dim=-1)
        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)
        return embedding


class LocationEncoder(nn.Module):
    """Sin-cos encoding of geographic coordinates.

    Each location is encoded from its latitude and longitude values.

    .. versionadded:: 0.11
    """

    def __init__(self, embed_dim: int, trainable_scale: bool = False) -> None:
        """Initialize a new LocationEncoder instance.

        Args:
            embed_dim: Output embedding dimension.
            trainable_scale: Learn a global scaling factor for the embeddings.
        """
        super().__init__()
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: Tensor) -> Tensor:
        """Encode the geographic coordinates.

        Args:
            location_coords: Latitude and longitude values of shape (B, 2).

        Returns:
            Location embeddings of shape (B, 1, embed_dim).
        """
        b = location_coords.shape[0]
        lat = _get_1d_sincos_pos_embed(
            self.lat_embed_dim, location_coords[:, 0].flatten()
        ).reshape(b, 1, -1)
        lon = _get_1d_sincos_pos_embed(
            self.lon_embed_dim, location_coords[:, 1].flatten()
        ).reshape(b, 1, -1)
        return self.scale * torch.cat([lat, lon], dim=-1)


class Prithvi(nn.Module):
    """Prithvi-EO-2.0 vision transformer encoder.

    Corresponds to the encoder of the pretrained Prithvi-EO-2.0 models
    found in the IBM-NASA geospatial repository:

    * https://huggingface.co/ibm-nasa-geospatial

    Encoder-only implementation of the multi-temporal masked autoencoder
    from `"Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for
    Geospatial Applications" <https://arxiv.org/abs/2412.02732>`_. The class
    token is stored at index 0 of each output feature map. Use
    :meth:`prepare_features_for_image_model` to reshape the temporal token
    sequence into spatial feature maps for 2D decoders.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2412.02732

    .. versionadded:: 0.11
    """

    pos_embed: Tensor
    cls_token: Tensor
    patch_embed: PatchEmbed3D

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        num_frames: int = 4,
        in_chans: int = 6,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        """Initialize a new Prithvi instance.

        Args:
            img_size: Input image size in pixels.
            patch_size: Patch size as (time, height, width) in pixels.
            num_frames: Number of input frames per sample.
            in_chans: Number of input image channels.
            embed_dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads per block.
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
            norm_layer: Normalization layer used by the transformer blocks.
            coords_encoding: Temporal and/or location encodings to enable,
                from {'time', 'location'}.
            coords_scale_learn: Learn a global scaling factor for the
                coordinate encodings.
            drop_path: Stochastic depth rate per block.
        """
        super().__init__()
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed3D(
            input_size=(num_frames, *self.img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.out_channels = [embed_dim * self.patch_embed.grid_size[0]] * depth

        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            'pos_embed', torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )

        self.blocks = nn.ModuleList(
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                drop_path=drop_path,
            )
            for _ in range(depth)
        )
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the position embeddings and transformer weights."""
        pos_embed = _get_3d_sincos_pos_embed(
            self.embed_dim, self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(pos_embed.to(self.pos_embed.dtype).unsqueeze(0))

        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(
        self,
        x: Tensor,
        temporal_coords: Tensor | None = None,
        location_coords: Tensor | None = None,
    ) -> list[Tensor]:
        """Compute features for the input images.

        Args:
            x: Input images of shape (B, C, T, H, W), or (B, C, H, W) when
                *num_frames* is 1.
            temporal_coords: Optional year and day-of-year values of shape
                (B, T, 2).
            location_coords: Optional latitude and longitude values of shape
                (B, 2).

        Returns:
            A list of ``depth`` feature maps of shape (B, 1 + T' * H' * W',
            embed_dim), one per transformer block. The last feature map is
            normalized.
        """
        if x.ndim == 4 and self.num_frames == 1:
            x = x.unsqueeze(2)
        t, h, w = x.shape[-3:]
        tokens_per_frame = (h // self.patch_size[1]) * (w // self.patch_size[2])

        pos_embed = _interpolate_pos_embedding(
            self.pos_embed,
            self.patch_embed.grid_size,
            self.patch_size,
            (t, h, w),
            self.embed_dim,
        )
        x = self.patch_embed(x)
        x = x + pos_embed[:, 1:]

        if self.temporal_encoding and temporal_coords is not None:
            x = x + self.temporal_embed_enc(temporal_coords, tokens_per_frame)
        if self.location_encoding and location_coords is not None:
            x = x + self.location_embed_enc(location_coords)

        cls_token = self.cls_token + pos_embed[:, :1]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        features[-1] = self.norm(x)
        return features

    def prepare_features_for_image_model(self, features: list[Tensor]) -> list[Tensor]:
        """Reshape temporal features into spatial feature maps.

        Collapses the time dimension into the channel dimension so the
        features can be used with 2D decoders.

        Args:
            features: Feature maps returned by :meth:`forward`.

        Returns:
            A list of feature maps of shape (B, T' * embed_dim, H', W').
        """
        time_dim = self.num_frames // self.patch_size[0]
        out = []
        for x in features:
            x = x[:, 1:]
            size = math.isqrt(x.shape[1] // time_dim)
            out.append(
                rearrange(
                    x,
                    'b (t h w) e -> b (t e) h w',
                    e=self.embed_dim,
                    t=time_dim,
                    h=size,
                )
            )
        return out


class PrithviV2_Weights(WeightsEnum):
    """Prithvi-EO-2.0 weights.

    .. versionadded:: 0.11
    """

    EO_V2_300 = Weights(
        url='https://huggingface.co/ModarIbrahim/prithvi-weights/resolve/0fe13e1b70dee7d3bfbe513ce246938d975af50f/prithvi_eo_v2_300-faab2c8b.pt',
        transforms=_prithvi_transforms,
        meta={
            'dataset': 'HLS',
            'model': 'prithvi_eo_v2_300',
            'publication': 'https://arxiv.org/abs/2412.02732',
            'repo': 'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M',
            'license': 'Apache-2.0',
            'ssl_method': 'MAE',
            'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07'],
            'in_chans': 6,
            'img_size': 224,
            'num_frames': 4,
            'coords_encoding': [],
        },
    )


def prithvi_eo_v2_300(
    weights: PrithviV2_Weights | None = None, *args: Any, **kwargs: Any
) -> Prithvi:
    """Prithvi-EO-2.0 300M model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2412.02732

    .. versionadded:: 0.11

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Prithvi`.
        **kwargs: Additional keyword arguments to pass to :class:`Prithvi`.

    Returns:
        A Prithvi-EO-2.0 300M model.
    """
    coords_encoding: list[str] = []
    if weights:
        coords_encoding = weights.meta['coords_encoding']

    model = Prithvi(
        *args,
        img_size=224,
        num_frames=4,
        patch_size=(1, 16, 16),
        in_chans=6,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        coords_encoding=coords_encoding,
        **kwargs,
    )

    if weights:
        state_dict = weights.get_state_dict(
            progress=True, check_hash=True, weights_only=True
        )
        # The positional embeddings are deterministic sin-cos values, so they
        # are recomputed at construction rather than loaded (the same
        # convention as the upstream checkpoint).
        state_dict = {
            key.removeprefix('encoder.'): value
            for key, value in state_dict.items()
            if key.startswith('encoder.') and 'pos_embed' not in key
        }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert not unexpected_keys
        assert set(missing_keys) <= {'pos_embed'}

    return model
