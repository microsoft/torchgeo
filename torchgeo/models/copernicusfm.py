# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# https://github.com/zhu-xlab/Copernicus-FM

"""Copernicus Foundation Model (Copernicus-FM)."""

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from timm.models.vision_transformer import Block
from torch import Tensor, vmap
from torchvision.models._api import Weights, WeightsEnum

from ..samplers.utils import _to_tuple
from .dofa import FCResLayer, TransformerWeightGenerator


def resize_abs_pos_embed(
    pos_embed: Tensor,
    new_size: int | tuple[int, int],
    old_size: int | tuple[int, int],
    num_prefix_tokens: int = 1,
    interpolation: str = 'bicubic',
    antialias: bool = True,
) -> Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation.

    Borrowed from https://github.com/bwconrad/flexivit. Copyright (c) 2023 Ben Conrad.

    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (e.g., cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing

    Returns:
        Resized pos_embed of size [b, n', d]
    """
    new_size = _to_tuple(new_size)
    old_size = _to_tuple(old_size)
    new_ntok = new_size[0] * new_size[1]

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed


def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: tuple[int, int],
    interpolation: str = 'bicubic',
    antialias: bool = True,
) -> Tensor:
    """Resample patch embeddings to a target resolution via pseudo-inverse resizing.

    Borrowed from https://github.com/bwconrad/flexivit. Copyright (c) 2023 Ben Conrad.

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing

    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, 'Patch embed kernel should be a 4D tensor'
    assert len(new_patch_size) == 2, 'New patch size should only be (height, width)'

    _, _, h, w = patch_embed.shape
    old_patch_size = (h, w)

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: tuple[int, int]) -> Tensor:
        x = F.interpolate(
            x[None, None, ...], shape, mode=interpolation, antialias=antialias
        )
        return x[0, 0, ...]

    def calculate_pinv(
        old_shape: tuple[int, int], new_shape: tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        pinv: Tensor = torch.linalg.pinv(resize_matrix)
        return pinv

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor) -> Tensor:
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, '(h w) -> h w', h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    patch_embed = v_resample_patch_embed(patch_embed)
    return patch_embed


class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Borrowed from https://github.com/microsoft/aurora.
    Copyright (c) Microsoft Corporation.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower: Lower wavelength.
            upper: Upper wavelength.
            assert_range: Assert that the encoded tensor is within the specified
                wavelength range.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: Tensor, d: int) -> Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x: Input to expand of shape `(..., n)`. All elements of `x` must
               lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d: Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x`
                are not within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(
            self.lower <= x.abs(), torch.all(x.abs() <= self.upper)
        )
        # Allow zeros to pass through.
        in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f'The input tensor is not within the configured range'
                f' `[{self.lower}, {self.upper}]`.'
            )

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError('The dimensionality must be a multiple of two.')

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum('...i,j->...ij', x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.


class DynamicPatchEmbed(nn.Module):
    """Dynamic patch embedding with spectral or variable hypernetworks.

    Adapted from DOFA.
    """

    def __init__(
        self,
        wv_planes: int = 128,
        kernel_size: int = 16,
        embed_dim: int = 1024,
        hypernet: Literal['spectral', 'variable'] = 'spectral',
    ) -> None:
        """Initialize a new DynamicPatchEmbed instance.

        Args:
            wv_planes: dim for wavelength/bandwidth/varname encoding.
            kernel_size: Kernel size for the patch embedding (convolution) layer.
            embed_dim: Embedding dimension.
            hypernet: Type of hypernetwork to use. Options: 'spectral' or 'variable'.
                'spectral' uses Fourier encodings for wavelength and bandwidth;
                'variable' uses a language embedding for variable names.
        """
        super().__init__()
        self.hypernet = hypernet
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        if self.hypernet == 'spectral':
            # Spectral hypernetwork: Fourier encoding for wavelength and bandwidth.
            # min wavelength: ultraviolet light (100 nm)
            # max wavelength: radio waves (1 m)
            # min bandwidth: s2 ~ 10 nm
            # max bandwidth: s1 ~ 1 m
            self.spectrum_central_expansion = FourierExpansion(100, 1e9)
            self.spectrum_bandwidth_expansion = FourierExpansion(1, 1e9)
        elif self.hypernet == 'variable':
            # Variable hypernetwork: Language embedding for variable names.
            self.language_proj = nn.Linear(2048, self.wv_planes)

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    def _get_weights(self, waves: Tensor) -> Tensor:
        """Use the dynamic weight generator.

        Args:
            waves: Spectral wavelengths.

        Returns:
            Dynamic weights.
        """
        dynamic_weights: Tensor = self.weight_generator(waves)
        return dynamic_weights

    def weight_init(self, m: object) -> None:
        """Initialize weights of a single layer.

        Args:
            m: A single layer.
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize weights of all layers."""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(
        self,
        img_feat: Tensor,
        wvs: Tensor | None = None,
        bandwidths: Tensor | None = None,
        language_embed: Tensor | None = None,
        kernel_size: int | None = None,
    ) -> Tensor:
        """Forward pass.

        For hypernet=='spectral', `wvs` and `bandwidths` must be provided.
        For hypernet=='variable', `language_embed` must be provided.

        Args:
            img_feat: Input image tensor (B, C, H, W).
            wvs: Wavelengths in nm (required if hypernet=='spectral').
            bandwidths: Bandwidths in nm (required if hypernet=='spectral').
            language_embed: Language embedding tensor from Llama 3.2 1B (length 2048).
            kernel_size: If provided and differs from the initialized kernel size,
                the generated patch embed kernel weights are resized accordingly.

        Returns:
            Output after patch embedding (B, N, D).

        Raises:
            ValueError: When *hypernet=='spectral'* and *wvs* or *bandwidths* is missing,
                or when *hypernet=='variable'* and *language_embed* is missing.
        """
        if self.hypernet == 'spectral':
            if wvs is None or bandwidths is None:
                msg = 'For spectral hypernet, wvs and bandwidths must be provided.'
                raise ValueError(msg)

            emb_central = self.spectrum_central_expansion(wvs, self.wv_planes)
            emb_bandwidth = self.spectrum_bandwidth_expansion(
                bandwidths, self.wv_planes
            )
            waves = emb_central + emb_bandwidth
        elif self.hypernet == 'variable':
            if language_embed is None:
                msg = 'For variable hypernet, language_embed must be provided.'
                raise ValueError(msg)

            # Expand dims to match batch size.
            waves = self.language_proj(language_embed.unsqueeze(0))

        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)
        inplanes = waves.size(0)
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute(3, 0, 1, 2)

        if kernel_size is not None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(
                dynamic_weight, (kernel_size, kernel_size)
            )
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view(self.embed_dim) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1
        )
        x = dynamic_out.flatten(2).transpose(1, 2)
        return x


class CopernicusFM(nn.Module):
    """CopernicusFM: VisionTransformer backbone.

    Example:
        **1. Spectral Mode (Using Wavelength and Bandwidth):**

        >>> model = CopernicusFM()
        >>> img = torch.randn(1, 4, 224, 224) # input image
        >>> meta = torch.full((1, 4), float('nan')) # [lon (degree), lat (degree), delta_time (days since 1970/1/1), patch_token_area (km^2)], assume unknown
        >>> wvs = [490, 560, 665, 842] # wavelength (nm): B,G,R,NIR (Sentinel 2)
        >>> bws = [65, 35, 30, 115] # bandwidth (nm): B,G,R,NIR (Sentinel 2)
        >>> kernel_size = 16 # expected patch size
        >>> input_mode = 'spectral'
        >>> logit = model(img, meta, wave_list=wvs, bandwidth=bws, input_mode=input_mode, kernel_size=kernel_size)
        >>> print(logit.shape)

        **2. Variable Mode (Using language embedding):**

        >>> model = CopernicusFM()
        >>> varname = 'Sentinel 5P Nitrogen Dioxide' # variable name (as input to a LLM for langauge embed)
        >>> img = torch.randn(1, 1, 56, 56) # input image
        >>> meta = torch.full((1, 4), float('nan')) # [lon (degree), lat (degree), delta_time (days since 1970/1/1), patch_token_area (km^2)], assume unknown
        >>> language_embed = torch.randn(2048) # language embedding: encode varname with a LLM (e.g. Llama)
        >>> kernel_size = 4 # expected patch size
        >>> input_mode = 'variable'
        >>> logit = model(img, meta, language_embed=language_embed, input_mode=input_mode, kernel_size=kernel_size)
        >>> print(logit.shape)

    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        drop_rate: float = 0.0,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        wv_planes: int = 128,
        num_classes: int = 0,
        global_pool: bool = True,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """Initialize a new CopernicusFM instance.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            drop_rate: Head dropout rate.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            wv_planes: Dimensions of dynamic weight generator.
            num_classes: Number of classes for classification head.
            global_pool: Whether or not to perform global pooling.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            norm_layer: Normalization layer.
        """
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed_spectral = DynamicPatchEmbed(
            wv_planes=128, kernel_size=16, embed_dim=embed_dim, hypernet='spectral'
        )
        self.patch_embed_variable = DynamicPatchEmbed(
            wv_planes=128, kernel_size=16, embed_dim=embed_dim, hypernet='variable'
        )

        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed sin-cos embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )

        self.coord_expansion = FourierExpansion(0.0001, 720)
        self.scale_expansion = FourierExpansion(0.001, 5.1e8)  # 1m2 to 5.1e8 km2
        # 1 to 365.25 days, enable more than 1 year
        self.time_expansion = FourierExpansion(1, 365.25, assert_range=False)
        self.coord_fc = nn.Linear(embed_dim, embed_dim)
        self.scale_fc = nn.Linear(embed_dim, embed_dim)
        self.time_fc = nn.Linear(embed_dim, embed_dim)
        # if meta info is not available, set to a learned parameter
        self.coord_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def get_coord_pos_embed(self, lons: Tensor, lats: Tensor, embed_dim: int) -> Tensor:
        """Geospatial coordinate position embedding.

        Args:
            lons: Longitudes (x).
            lats: Latitudes (y).
            embed_dim: Embedding dimension.

        Returns:
            Coordinate position embedding.
        """
        coord_embed_lon = self.coord_expansion(lons + 180, embed_dim // 2)
        coord_embed_lat = self.coord_expansion(lats + 90, embed_dim // 2)
        coord_embed = torch.cat([coord_embed_lon, coord_embed_lat], dim=-1)

        if coord_embed.shape[-1] < embed_dim:
            # pad zeros
            coord_embed = torch.cat(
                (
                    coord_embed,
                    torch.zeros(
                        coord_embed.shape[0],
                        embed_dim - coord_embed.shape[-1],
                        device=coord_embed.device,
                    ),
                ),
                dim=-1,
            )

        return coord_embed.unsqueeze(1)  # [B,1,D]

    def get_area_pos_embed(self, areas: Tensor, embed_dim: int) -> Tensor:
        """Geospatial area position embedding.

        Args:
            areas: Spatial areas.
            embed_dim: Embedding dimension.

        Returns:
            Area position embedding.
        """
        scale_embed: Tensor = self.scale_expansion(areas, embed_dim)  # B, D
        scale_embed = scale_embed.unsqueeze(1)  # [B,1,D]
        return scale_embed

    def get_time_pos_embed(self, times: Tensor, embed_dim: int) -> Tensor:
        """Geotemporal position embedding.

        Args:
            times: Timestamps.
            embed_dim: Embedding dimension.

        Returns:
            Temporal position embedding.
        """
        time_embed: Tensor = self.time_expansion(times, embed_dim)  # B, D
        time_embed = time_embed.unsqueeze(1)  # [B,1,D]
        return time_embed

    def forward_features(
        self,
        x: Tensor,
        meta_info: Tensor,
        wave_list: Sequence[float] | None = None,
        bandwidth: Sequence[float] | None = None,
        language_embed: Tensor | None = None,
        input_mode: Literal['spectral', 'variable'] = 'spectral',
        kernel_size: int | None = None,
    ) -> Tensor:
        """Forward pass of the feature embedding layer.

        Args:
            x: Input mini-batch.
            meta_info: Longitudes (degree), latitudes (degree), times
                (days since 1970/1/1), and areas (km^2) of each patch.
                Use NaN for unknown metadata.
            wave_list: Wavelengths of each spectral band (nm).
                Only used if *input_mode=='spectral'*.
            bandwidth: Bandwidths in nm.
                Only used if *input_mode=='spectral'*.
            language_embed: Language embedding tensor from Llama 3.2 1B (length 2048).
                Only used if *input_mode=='variable'*.
            input_mode: One of 'spectral' or 'variable'.
            kernel_size: If provided and differs from the initialized kernel size,
                the generated patch embed kernel weights are resized accordingly.

        Returns:
            Output mini-batch.
        """
        if input_mode == 'spectral':
            wvs = torch.tensor(wave_list, device=x.device).float()
            bandwidths = torch.tensor(bandwidth, device=x.device).float()
            x = self.patch_embed_spectral(
                x, wvs=wvs, bandwidths=bandwidths, kernel_size=kernel_size
            )
        elif input_mode == 'variable':
            x = self.patch_embed_variable(
                x, language_embed=language_embed, kernel_size=kernel_size
            )

        # resize pos embed
        num_patches = x.size(1)
        num_patches_sqrt = int(math.sqrt(num_patches))
        num_patches_sqrt_origin = int(math.sqrt(self.num_patches))
        pos_embed = resize_abs_pos_embed(
            self.pos_embed,
            num_patches_sqrt,
            (num_patches_sqrt_origin, num_patches_sqrt_origin),
            num_prefix_tokens=1,
        )

        # coord, scale and time pos embed
        lons, lats, times, areas = (
            meta_info[:, 0],
            meta_info[:, 1],
            meta_info[:, 2],
            meta_info[:, 3],
        )
        embed_dim = pos_embed.shape[-1]
        if torch.isnan(lons).any() or torch.isnan(lats).any():
            coord_embed: nn.Parameter | Tensor = self.coord_token
        else:
            coord_embed = self.get_coord_pos_embed(lons, lats, embed_dim)
        coord_embed = self.coord_fc(coord_embed)
        if torch.isnan(areas).any():
            area_embed: nn.Parameter | Tensor = self.scale_token
        else:
            area_embed = self.get_area_pos_embed(areas, embed_dim)
        area_embed = self.scale_fc(area_embed)
        if torch.isnan(times).any():
            time_embed: nn.Parameter | Tensor = self.time_token
        else:
            time_embed = self.get_time_pos_embed(times, embed_dim)
        time_embed = self.time_fc(time_embed)
        pos_embed = pos_embed + coord_embed + area_embed + time_embed

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome: Tensor = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the attention head.

        Args:
            x: Input mini-batch.
            pre_logits: Whether or not to return the layer before logits are computed.

        Returns:
            Output mini-batch.
        """
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Tensor,
        meta_info: Tensor,
        wave_list: Sequence[float] | None = None,
        bandwidth: Sequence[float] | None = None,
        language_embed: Tensor | None = None,
        input_mode: Literal['spectral', 'variable'] = 'spectral',
        kernel_size: int | None = None,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            meta_info: Longitudes (degree), latitudes (degree), times
                (days since 1970/1/1), and areas (km^2) of each patch.
                Use NaN for unknown metadata.
            wave_list: Wavelengths of each spectral band (nm).
                Only used if *input_mode=='spectral'*.
            bandwidth: Bandwidths in nm.
                Only used if *input_mode=='spectral'*.
            language_embed: Language embedding tensor from Llama 3.2 1B (length 2048).
                Only used if *input_mode=='variable'*.
            input_mode: One of 'spectral' or 'variable'.
            kernel_size: If provided and differs from the initialized kernel size,
                the generated patch embed kernel weights are resized accordingly.

        Returns:
            Output mini-batch.
        """
        fx = self.forward_features(
            x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size
        )
        x = self.forward_head(fx)
        return x


class CopernicusFM_Base_Weights(WeightsEnum):  # type: ignore[misc]
    """Copernicus-FM-base weights."""

    CopernicusFM_ViT = Weights(
        url='https://huggingface.co/torchgeo/copernicus-fm/resolve/f395812cc990ba25a451dbb9c9e6d95c8482947e/CopernicusFM_ViT_base_varlang-085350e4.pth',
        transforms=None,
        meta={
            'dataset': 'Copernicus-Pretrain',
            'model': 'copernicusfm_base',
            'publication': 'https://arxiv.org/abs/2503.11849',
            'repo': 'https://github.com/zhu-xlab/Copernicus-FM',
            'ssl_method': 'mae+distill',
        },
    )


def copernicusfm_base(
    weights: CopernicusFM_Base_Weights | None = None, *args: Any, **kwargs: Any
) -> CopernicusFM:
    """CopernicusFM vit-base model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.7

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`CopernicusFM`.
        **kwargs: Additional keyword arguments to pass to :class:`CopernicusFM`.

    Returns:
        A CopernicusFM base model.
    """
    kwargs |= {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    model = CopernicusFM(*args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )

        print(missing_keys)
        print(unexpected_keys)

        # Both fc_norm and head are generated dynamically
        assert set(missing_keys) <= {
            'fc_norm.weight',
            'fc_norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model
