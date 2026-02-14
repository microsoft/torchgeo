# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Copyright (c) 2025 Galileo Authors
#
# Modified from https://github.com/nasaharvest/galileo/blob/main/single_file_galileo.py

"""Galileo model implementation."""

import itertools
from collections.abc import Iterable, Sequence
from typing import Any, Final, cast

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor, vmap
from torchvision.models._api import Weights, WeightsEnum

PRETRAINING_NORMALIZING_DICT = {
    'space_time': {
        'mean': [
            -11.728724389184965,
            -18.85558188024017,
            1395.3408730676722,
            1338.4026921784578,
            1343.09883810357,
            1543.8607982512297,
            2186.2022069512263,
            2525.0932853316694,
            2410.3377187373408,
            2750.2854646886753,
            2234.911100061487,
            1474.5311266077113,
            0.2892116502999044,
        ],
        'std': [
            4.887145774840316,
            5.730270320384293,
            917.7041440370853,
            913.2988423581528,
            1092.678723527555,
            1047.2206083460424,
            1048.0101611156767,
            1143.6903026819996,
            1098.979177731649,
            1204.472755085893,
            1145.9774063078878,
            980.2429840007796,
            0.2720939024500081,
        ],
    },
    'space': {
        'mean': [
            673.0152819503361,
            5.930092668915115,
            0.10470439140978786,
            0.23965913270066183,
            0.08158044385860364,
            0.04246976254259546,
            0.11304392863520317,
            0.17329647890362473,
            0.0698981691616277,
            0.12130267132802142,
            0.04671318615236216,
            10.973119802517362,
            1.0927069179958768,
            1.6991394232855903,
            0.03720594618055555,
            1.3671352688259548,
        ],
        'std': [
            983.0697298296237,
            8.167406789813247,
            0.18771647977504985,
            0.2368313455675914,
            0.08024268534756586,
            0.04045374496146404,
            0.11350342472061795,
            0.1279898111718168,
            0.12042341550438586,
            0.13602408145504347,
            0.043971116096060345,
            31.255340146970997,
            10.395974878206689,
            12.92380617159917,
            1.9285254295940466,
            11.612179775408928,
        ],
    },
    'time': {
        'mean': [
            271.5674963541667,
            0.08554303677156568,
            657.3181260091111,
            692.1291795806885,
            562.781331880633,
            1.5647115934036673,
        ],
        'std': [
            79.80828940314429,
            0.11669547098151486,
            704.0008695557707,
            925.0116126406431,
            453.2434022278578,
            7.513020170832818,
        ],
    },
    'static': {
        'mean': [
            188.20315880851746,
            0.2804946561574936,
            0.11371652073860168,
            0.058778801321983334,
            0.10474256777763366,
            0.2396918488264084,
            0.08152248692512512,
            0.04248040814399719,
            0.11303179881572724,
            0.17326324067115784,
            0.06998309404850006,
            0.12122812910079957,
            0.04671641788482666,
            10.98456594619751,
            1.0968475807189941,
            1.6947754135131836,
            0.03320046615600586,
            1.3602827312469483,
        ],
        'std': [
            1154.5919128300602,
            0.5276998078079327,
            0.7021637331734328,
            0.36528892213195063,
            0.17470213191865785,
            0.20411195416718833,
            0.0660782470089761,
            0.03380702424871257,
            0.09809195568521663,
            0.11292471052124119,
            0.09720748930233268,
            0.12912217763726777,
            0.0399973913151906,
            23.725471823867462,
            5.715238079725388,
            9.030481416228302,
            0.9950220242487364,
            7.754429123862099,
        ],
    },
}

BASE_GSD = 10
DEFAULT_MONTH = 5

# band information
S1_BANDS = ['VV', 'VH']
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
ERA5_BANDS = ['temperature_2m', 'total_precipitation_sum']
TC_BANDS = ['def', 'soil', 'aet']
VIIRS_BANDS = ['avg_rad']
SRTM_BANDS = ['elevation', 'slope']
DW_BANDS = [
    'DW_water',
    'DW_trees',
    'DW_grass',
    'DW_flooded_vegetation',
    'DW_crops',
    'DW_shrub_and_scrub',
    'DW_built',
    'DW_bare',
    'DW_snow_and_ice',
]
WC_BANDS = [
    'WC_temporarycrops',
    'WC_maize',
    'WC_wintercereals',
    'WC_springcereals',
    'WC_irrigation',
]
STATIC_DW_BANDS = [f'{x}_static' for x in DW_BANDS]
STATIC_WC_BANDS = [f'{x}_static' for x in WC_BANDS]

LANDSCAN_BANDS = ['b1']
LOCATION_BANDS = ['x', 'y', 'z']

SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ['NDVI']
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS


SPACE_TIME_BANDS_GROUPS_IDX = {
    'S1': [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
    'S2_RGB': [SPACE_TIME_BANDS.index(b) for b in ['B2', 'B3', 'B4']],
    'S2_Red_Edge': [SPACE_TIME_BANDS.index(b) for b in ['B5', 'B6', 'B7']],
    'S2_NIR_10m': [SPACE_TIME_BANDS.index(b) for b in ['B8']],
    'S2_NIR_20m': [SPACE_TIME_BANDS.index(b) for b in ['B8A']],
    'S2_SWIR': [SPACE_TIME_BANDS.index(b) for b in ['B11', 'B12']],
    'NDVI': [SPACE_TIME_BANDS.index('NDVI')],
}

TIME_BAND_GROUPS_IDX = {
    'ERA5': [TIME_BANDS.index(b) for b in ERA5_BANDS],
    'TC': [TIME_BANDS.index(b) for b in TC_BANDS],
    'VIIRS': [TIME_BANDS.index(b) for b in VIIRS_BANDS],
}

SPACE_BAND_GROUPS_IDX = {
    'SRTM': [SPACE_BANDS.index(b) for b in SRTM_BANDS],
    'DW': [SPACE_BANDS.index(b) for b in DW_BANDS],
    'WC': [SPACE_BANDS.index(b) for b in WC_BANDS],
}

STATIC_BAND_GROUPS_IDX = {
    'LS': [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
    'location': [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
    'DW_static': [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
    'WC_static': [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
}


def to_cartesian(lat: Tensor, lon: Tensor) -> Tensor:
    """Convert lat/lon in degrees to Cartesian coordinates.

    Args:
        lat: Latitude value in degrees.
        lon: Longitude value in degrees.

    Returns:
        Cartesian tensor ``[3]`` with ``(x, y, z)`` coordinates.
    """
    if torch.any(lat < -90) or torch.any(lat > 90):
        raise ValueError('Latitude must be in [-90, 90] degrees')
    if torch.any(lon < -180) or torch.any(lon > 180):
        raise ValueError('Longitude must be in [-180, 180] degrees')
    lat = torch.deg2rad(lat)
    lon = torch.deg2rad(lon)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def _normalize_input(x: Tensor, key: str, channel_indices: list[int]) -> Tensor:
    """Normalize a provided input tensor over its last dimension.

    Args:
        x: Input tensor with channel dimension last.
        key: Normalization key in ``PRETRAINING_NORMALIZING_DICT``.
        channel_indices: Channel indices in the full modality tensor.

    Returns:
        Normalized tensor with the same shape as ``x``.
    """
    stats = PRETRAINING_NORMALIZING_DICT[key]
    mean = torch.tensor(
        [stats['mean'][idx] for idx in channel_indices], device=x.device, dtype=x.dtype
    )
    std = torch.tensor(
        [stats['std'][idx] for idx in channel_indices], device=x.device, dtype=x.dtype
    )
    return (x - mean) / std


def construct_inputs(
    s1: torch.Tensor | None = None,  # [H, W, T, D]
    s2: torch.Tensor | None = None,  # [H, W, T, D]
    era5: torch.Tensor | None = None,  # [T, D]
    tc: torch.Tensor | None = None,  # [T, D]
    viirs: torch.Tensor | None = None,  # [T, D]
    srtm: torch.Tensor | None = None,  # [H, W, D]
    dw: torch.Tensor | None = None,  # [H, W, D]
    wc: torch.Tensor | None = None,  # [H, W, D]
    landscan: torch.Tensor | None = None,  # [D]
    latlon: torch.Tensor | None = None,  # [D]
    months: torch.Tensor | None = None,  # [T]
    normalize: bool = False,
) -> dict[str, Tensor]:
    """Construct Galileo inputs and binary masks from modality tensors.

    Args:
        s1: Sentinel-1 tensor of shape ``[H, W, T, 2]``.
        s2: Sentinel-2 tensor of shape ``[H, W, T, 10]``.
        era5: ERA5 tensor of shape ``[T, 2]``.
        tc: TerraClimate tensor of shape ``[T, 3]``.
        viirs: VIIRS tensor of shape ``[T, 1]``.
        srtm: SRTM tensor of shape ``[H, W, 2]``.
        dw: Dynamic World tensor of shape ``[H, W, 9]``.
        wc: WorldCereal tensor of shape ``[H, W, 5]``.
        landscan: LandScan tensor of shape ``[1]``.
        latlon: Latitude/longitude tensor of shape ``[2]`` in EPSG:4326.
        months: Month indices of shape ``[T]``.
        normalize: Whether to apply mean/std normalization per modality group.

    Returns:
        Dictionary with keys matching :meth:`Encoder.forward` modality/mask inputs.
    """
    space_time_inputs = [s1, s2]
    time_inputs = [era5, tc, viirs]
    space_inputs = [srtm, dw, wc]
    static_inputs = [landscan, latlon]
    devices = [
        x.device
        for x in space_time_inputs + time_inputs + space_inputs + static_inputs
        if x is not None
    ]

    if len(devices) == 0:
        raise ValueError('At least one input must be not None')
    if not all(devices[0] == device for device in devices):
        raise ValueError('Received tensors on multiple devices')
    device = devices[0]

    # first, check all the input shapes are consistent
    timesteps_list = [x.shape[2] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in time_inputs if x is not None
    ]
    height_list = [x.shape[0] for x in space_time_inputs if x is not None] + [
        x.shape[0] for x in space_inputs if x is not None
    ]
    width_list = [x.shape[1] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in space_inputs if x is not None
    ]

    if len(timesteps_list) > 0:
        if not all(timesteps_list[0] == timestep for timestep in timesteps_list):
            raise ValueError('Inconsistent number of timesteps per input')
        t = timesteps_list[0]
    else:
        t = 1

    if len(height_list) > 0:
        if not all(height_list[0] == height for height in height_list):
            raise ValueError('Inconsistent heights per input')
        if not all(width_list[0] == width for width in width_list):
            raise ValueError('Inconsistent widths per input')
        h = height_list[0]
        w = width_list[0]
    else:
        h, w = 1, 1

    # now, we can construct our empty input tensors. By default, everything is masked
    s_t_x = torch.zeros(
        (h, w, t, len(SPACE_TIME_BANDS)), dtype=torch.float, device=device
    )
    s_t_m = torch.ones(
        (h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), dtype=torch.float, device=device
    )
    sp_x = torch.zeros((h, w, len(SPACE_BANDS)), dtype=torch.float, device=device)
    sp_m = torch.ones(
        (h, w, len(SPACE_BAND_GROUPS_IDX)), dtype=torch.float, device=device
    )
    t_x = torch.zeros((t, len(TIME_BANDS)), dtype=torch.float, device=device)
    t_m = torch.ones((t, len(TIME_BAND_GROUPS_IDX)), dtype=torch.float, device=device)
    st_x = torch.zeros((len(STATIC_BANDS)), dtype=torch.float, device=device)
    st_m = torch.ones((len(STATIC_BAND_GROUPS_IDX)), dtype=torch.float, device=device)

    for x, bands_list, group_key in zip([s1, s2], [S1_BANDS, S2_BANDS], ['S1', 'S2']):
        if x is not None:
            indices = [
                idx for idx, val in enumerate(SPACE_TIME_BANDS) if val in bands_list
            ]
            groups_idx = [
                idx
                for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX)
                if group_key in key
            ]
            if normalize:
                x = _normalize_input(x, 'space_time', indices)
            s_t_x[:, :, :, indices] = x
            s_t_m[:, :, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [srtm, dw, wc], [SRTM_BANDS, DW_BANDS, WC_BANDS], ['SRTM', 'DW', 'WC']
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(SPACE_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(SPACE_BAND_GROUPS_IDX) if group_key in key
            ]
            if normalize:
                x = _normalize_input(x, 'space', indices)
            sp_x[:, :, indices] = x
            sp_m[:, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [era5, tc, viirs], [ERA5_BANDS, TC_BANDS, VIIRS_BANDS], ['ERA5', 'TC', 'VIIRS']
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(TIME_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(TIME_BAND_GROUPS_IDX) if group_key in key
            ]
            if normalize:
                x = _normalize_input(x, 'time', indices)
            t_x[:, indices] = x
            t_m[:, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [landscan, latlon], [LANDSCAN_BANDS, LOCATION_BANDS], ['LS', 'location']
    ):
        if x is not None:
            if group_key == 'location':
                # transform latlon to cartesian
                x = to_cartesian(x[0], x[1])
            indices = [idx for idx, val in enumerate(STATIC_BANDS) if val in bands_list]
            groups_idx = [
                idx
                for idx, key in enumerate(STATIC_BAND_GROUPS_IDX)
                if group_key in key
            ]
            if normalize:
                x = _normalize_input(x, 'static', indices)
            st_x[indices] = x
            st_m[groups_idx] = 0

    if months is None:
        months = torch.ones((t,), dtype=torch.long, device=device) * DEFAULT_MONTH
    else:
        if months.shape[0] != t:
            raise ValueError('Incorrect number of input months')

    return {
        'space_time_x': s_t_x,
        'space_time_mask': s_t_m,
        'space_x': sp_x,
        'space_mask': sp_m,
        'time_x': t_x,
        'time_mask': t_m,
        'static_x': st_x,
        'static_mask': st_m,
        'months': months,
    }


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int,
    grid_size: int,
    res: Tensor,
    cls_token: bool = False,
    device: str | torch.device = 'cpu',
) -> Tensor:
    """Compute 2D sine-cosine embeddings at arbitrary spatial resolutions.

    Args:
        embed_dim: Token embedding dimension.
        grid_size: Spatial grid size for both height and width.
        res: Resolution scaling tensor of shape ``(n,)``.
        cls_token: Whether to prepend a class-token embedding.
        device: Device for the generated tensors.

    Returns:
        Positional embeddings of shape ``(n, h*w, embed_dim)`` or
        ``(n, 1+h*w, embed_dim)`` if ``cls_token`` is enabled.
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid_xy = torch.meshgrid(
        grid_w, grid_h, indexing='xy'
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid_xy, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum('chw,n->cnhw', grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], device=pos_embed.device), pos_embed], dim=1
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor) -> Tensor:
    """Compute 2D sine-cosine embeddings from a coordinate grid.

    Args:
        embed_dim: Embedding dimension ``D``.
        grid: Coordinate tensor of shape ``[2, N, H, W]``.

    Returns:
        Positional embedding tensor of shape ``[N*H*W, D]``.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute 1D sine-cosine embeddings for scalar positions.

    Args:
        embed_dim: Embedding dimension ``D``.
        pos: Position tensor broadcastable to ``[M]``.

    Returns:
        Positional embedding tensor of shape ``[M, D]``.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_month_encoding_table(embed_dim: int) -> Tensor:
    """Return sinusoidal month encodings for months indexed 0-11.

    Args:
        embed_dim: Embedding dimension ``D``.

    Returns:
        Month embedding table of shape ``[12, D]``.
    """
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], dim=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], dim=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], dim=-1)

    return month_table  # (M, D)


# thanks to https://github.com/bwconrad/flexivit/ for this nice implementation
# of the FlexiPatchEmbed module
def to_2tuple(x: Any) -> tuple[Any, ...]:
    """Convert scalar input to a 2-tuple, preserving iterables.

    Args:
        x: Scalar or iterable value.

    Returns:
        A tuple with two elements for scalar inputs, else ``tuple(x)``.
    """
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))


class FlexiPatchEmbed(nn.Module):
    """Flexible patch embedding layer with runtime patch-size resizing."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6),
        interpolation: str = 'bicubic',
        antialias: bool = True,
    ) -> None:
        """Convert image tensors to patch embeddings with flexible patch sizes.

        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            patch_size: Base patch size. i.e the size of the parameter buffer
            in_chans: Number of input image channels
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            patch_size_seq: List of patch sizes to randomly sample from
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing

        Returns:
            None.
        """
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict[tuple[int, int], Tensor]:
        """Precompute pseudo-inverse matrices for configured patch sizes.

        Args:
            None.

        Returns:
            Mapping from patch size ``(h, w)`` to pseudo-inverse resize matrices.
        """
        pinvs: dict[tuple[int, int], Tensor] = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """Resize a 2D kernel basis tensor.

        Args:
            x: Input tensor of shape ``[H, W]``.
            shape: Output spatial shape ``(new_h, new_w)``.

        Returns:
            Resized tensor of shape ``[new_h, new_w]``.
        """
        x_resized = torch.nn.functional.interpolate(
            x[None, None, ...], shape, mode=self.interpolation, antialias=self.antialias
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: tuple[int, int], new_shape: tuple[int, int]
    ) -> Tensor:
        """Compute pseudo-inverse resize matrix for kernel resampling.

        Args:
            old_shape: Original kernel shape ``(h, w)``.
            new_shape: Target kernel shape ``(h, w)``.

        Returns:
            Pseudo-inverse resize matrix of shape ``[(new_h*new_w), (old_h*old_w)]``.
        """
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return cast(Tensor, torch.linalg.pinv(resize_matrix))

    def resize_patch_embed(
        self, patch_embed: Tensor, new_patch_size: tuple[int, int]
    ) -> Tensor:
        """Resize convolution kernels to a new patch size.

        Args:
            patch_embed: Kernel tensor of shape ``[out_ch, in_ch, h, w]``.
            new_patch_size: Target patch size ``(h, w)``.

        Returns:
            Resized kernel tensor of shape ``[out_ch, in_ch, new_h, new_w]``.
        """
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                self.patch_size, new_patch_size
            )
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor) -> Tensor:
            """Resample a single kernel using pseudo-inverse projection.

            Args:
                patch_embed: Kernel tensor of shape ``[h, w]``.

            Returns:
                Resampled kernel tensor of shape ``[new_h, new_w]``.
            """
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, '(h w) -> h w', h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return cast(Tensor, v_resample_patch_embed(patch_embed))

    def forward(
        self, x: Tensor, patch_size: int | tuple[int, int] | None = None
    ) -> Tensor:
        """Project inputs to patch embeddings for a chosen patch size.

        Args:
            x: Tensor of shape ``[B, H, W, C]`` or ``[B, H, W, T, C]``.
            patch_size: Optional patch size override.

        Returns:
            Embedded tensor of shape ``[B, H', W', D]`` or ``[B, H', W', T, D]``.
        """
        # x has input shape [b, h, w, (t), c]
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0  # ignored if has_time_dimension is False
        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, 'b h w t c -> (b t) c h w')
        else:
            x = rearrange(x, 'b h w c -> b c h w')

        if not patch_size:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
        # Apply conv with resized weights
        x = torch.nn.functional.conv2d(
            x, weight, bias=self.proj.bias, stride=patch_size
        )

        if has_time_dimension:
            x = rearrange(x, '(b t) c h w -> b h w t c', b=batch_size, t=num_timesteps)
        else:
            x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)

        return x


class Attention(nn.Module):
    """Multi-head self/cross attention used in Galileo blocks."""

    fast_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        cross_attn: bool = False,
    ) -> None:
        """Initialize an attention layer.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether ``q``, ``k``, and ``v`` projections use bias.
            qk_norm: Whether to apply per-head normalization to ``q`` and ``k``.
            attn_drop: Attention dropout probability.
            proj_drop: Output projection dropout probability.
            norm_layer: Normalization layer type.
            cross_attn: Whether this block runs cross-attention.

        Returns:
            None.
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.cross_attn = cross_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: Tensor, y: Tensor | None = None, attn_mask: Tensor | None = None
    ) -> Tensor:
        """Run self-attention or cross-attention over input tokens.

        Args:
            x: Query tokens of shape ``[B, N, D]``.
            y: Key/value tokens of shape ``[B, M, D]`` for cross-attention.
            attn_mask: Optional mask of shape ``[B, M]``.

        Returns:
            Output token tensor of shape ``[B, N, D]``.
        """
        B, N, C = x.shape

        q = self.q(x)

        if y is None:
            assert not self.cross_attn
            k = self.k(x)
            v = self.v(x)
        else:
            assert self.cross_attn
            k = self.k(y)
            v = self.v(y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                # a value of True indicates that the element should take part in attention
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """Initialize the MLP projection stack.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden feature dimension.
            out_features: Output feature dimension.
            act_layer: Activation module type.
            bias: Whether linear layers use bias.
            drop: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Run MLP layers on token embeddings.

        Args:
            x: Input tensor of shape ``[B, N, D]``.

        Returns:
            Output tensor of shape ``[B, N, D]``.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    """Per-channel residual scaling module."""

    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        """Initialize learnable scale parameters.

        Args:
            dim: Channel dimension to scale.
            init_values: Initial scalar value for each channel.
            inplace: Whether to scale in-place.

        Returns:
            None.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale inputs element-wise using learned coefficients.

        Args:
            x: Input tensor of shape ``[..., D]``.

        Returns:
            Scaled tensor with shape ``[..., D]``.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Apply stochastic depth to an input tensor.

    Args:
        x: Input tensor of shape ``[B, ...]``.
        drop_prob: Probability of dropping residual paths.
        training: Whether module is in training mode.

    Returns:
        Tensor with the same shape as ``x``.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None) -> None:
        """Initialize stochastic depth drop probability.

        Args:
            drop_prob: Probability of dropping residual paths.

        Returns:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth during training.

        Args:
            x: Input tensor of shape ``[B, ...]``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        return drop_path(x, self.drop_prob or 0.0, self.training)


class Block(nn.Module):
    """Transformer block with attention and MLP sub-layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        cross_attn: bool = False,
    ) -> None:
        """Initialize a transformer block.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden-to-input ratio.
            qkv_bias: Whether attention projections use bias.
            qk_norm: Whether to normalize ``q``/``k`` per head.
            drop: Projection and MLP dropout probability.
            attn_drop: Attention dropout probability.
            drop_path: Stochastic depth probability.
            init_values: Optional LayerScale initialization value.
            act_layer: MLP activation layer type.
            norm_layer: Normalization layer type.
            cross_attn: Whether to run cross-attention.

        Returns:
            None.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x: Tensor, y: Tensor | None, attn_mask: Tensor | None) -> Tensor:
        """Apply attention and MLP updates to token embeddings.

        Args:
            x: Query tokens of shape ``[B, N, D]``.
            y: Optional key/value tokens of shape ``[B, M, D]``.
            attn_mask: Optional attention mask.

        Returns:
            Updated token tensor of shape ``[B, N, D]``.
        """
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), y, attn_mask)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ModuleListWithInit(nn.ModuleList):
    """ModuleList variant with shared linear layer initialization."""

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize linear layers with Xavier uniform weights.

        Args:
            m: Module instance to initialize.

        Returns:
            None.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GalileoBase(nn.Module):
    """Shared tokenization and encoding utilities for Galileo models."""

    cross_attn: bool

    def __init__(
        self,
        embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        base_patch_size: int = 4,
        use_channel_embs: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        """Initialize common Galileo backbone components.

        Args:
            embedding_size: Token embedding dimension.
            depth: Number of transformer blocks.
            mlp_ratio: MLP hidden-to-input ratio.
            num_heads: Number of attention heads.
            max_sequence_length: Maximum temporal sequence length.
            base_patch_size: Default spatial patch size.
            use_channel_embs: Whether channel embeddings are learnable.
            drop_path: Stochastic depth probability.

        Returns:
            None.
        """
        super().__init__()

        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        self.space_groups = SPACE_BAND_GROUPS_IDX
        self.time_groups = TIME_BAND_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        self.blocks = ModuleListWithInit(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    cross_attn=self.cross_attn,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

        self.max_sequence_length = max_sequence_length
        # we have 4 embeddings (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension. This will change soon anyway
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]
            month_tab, freeze=True
        )
        if use_channel_embs:
            args = {'requires_grad': True}
        else:
            args = {'requires_grad': False}
        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_TIME_BANDS_GROUPS_IDX), int(embedding_size * 0.25)),
            **args,
        )
        self.sp_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(TIME_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.st_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize linear layers with Xavier uniform weights.

        Args:
            m: Module instance to initialize.

        Returns:
            None.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
    ) -> tuple[Tensor, Tensor]:
        """Flatten grouped tokens and concatenate along sequence axis.

        Args:
            s_t_x: Space-time tokens of shape ``[B, H, W, T, Cst, D]``.
            sp_x: Space-only tokens of shape ``[B, H, W, Cs, D]``.
            t_x: Time-only tokens of shape ``[B, T, Ct, D]``.
            st_x: Static tokens of shape ``[B, Cstatic, D]``.
            s_t_m: Space-time masks of shape ``[B, H, W, T, Cst]``.
            sp_m: Space-only masks of shape ``[B, H, W, Cs]``.
            t_m: Time-only masks of shape ``[B, T, Ct]``.
            st_m: Static masks of shape ``[B, Cstatic]``.

        Returns:
            Tuple ``(x, m)`` where ``x`` has shape ``[B, N, D]`` and
            ``m`` has shape ``[B, N]``.
        """
        s_t_x = rearrange(s_t_x, 'b h w t c_g d -> b (h w t c_g) d')
        sp_x = rearrange(sp_x, 'b h w c_g d -> b (h w c_g) d')
        t_x = rearrange(t_x, 'b t c_g d -> b (t c_g) d')

        s_t_m = rearrange(s_t_m, 'b h w t c_g-> b (h w t c_g)')
        sp_m = rearrange(sp_m, 'b h w c_g-> b (h w c_g)')
        t_m = rearrange(t_m, 'b t c_g -> b (t c_g)')

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
        t_c_g: int,
        st_c_g: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split flattened tokens back into grouped tensor layouts.

        Args:
            x: Flattened token tensor of shape ``[B, N, D]``.
            h: Spatial token height.
            w: Spatial token width.
            t: Temporal token length.
            s_t_c_g: Number of space-time channel groups.
            sp_c_g: Number of space-only channel groups.
            t_c_g: Number of time-only channel groups.
            st_c_g: Number of static channel groups.

        Returns:
            Tuple ``(s_t_x, sp_x, t_x, st_x)`` with shapes
            ``[B, H, W, T, Cst, D]``, ``[B, H, W, Cs, D]``, ``[B, T, Ct, D]``,
            and ``[B, Cstatic, D]``.
        """
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g

        s_t_x = rearrange(
            x[:, :n_s_t_t], 'b (h w t c) d -> b h w t c d', h=h, w=w, t=t, c=s_t_c_g
        )
        sp_x = rearrange(
            x[:, n_s_t_t : -(n_t_t + st_c_g)],
            'b (h w c) d -> b h w c d',
            h=h,
            w=w,
            c=sp_c_g,
        )
        t_x = rearrange(
            x[:, -(n_t_t + st_c_g) : -st_c_g], 'b (t c) d -> b t c d', t=t, c=t_c_g
        )
        st_x = x[:, -st_c_g:]

        return s_t_x, sp_x, t_x, st_x

    def apply_encodings(
        self,
        s_t_x: Tensor,
        sp_x: Tensor,
        t_x: Tensor,
        st_x: Tensor,
        months: Tensor,
        patch_size: int | None,
        input_res: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Add channel, temporal, month, and spatial encodings.

        Args:
            s_t_x: Space-time tokens of shape ``[B, H, W, T, Cst, D]``.
            sp_x: Space-only tokens of shape ``[B, H, W, Cs, D]``.
            t_x: Time-only tokens of shape ``[B, T, Ct, D]``.
            st_x: Static tokens of shape ``[B, Cstatic, D]``.
            months: Month indices of shape ``[B, T]``.
            patch_size: Patch size used for tokenization.
            input_res: Input pixel resolution in meters.

        Returns:
            Encoded token tuple ``(s_t_x, sp_x, t_x, st_x)`` with unchanged shapes.
        """
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g = sp_x.shape[-2], t_x.shape[-2]
        st_c_g = st_x.shape[-2]

        s_t_channel = repeat(
            self.s_t_channel_embed, 'c_g d -> b h w t c_g d', b=b, h=h, w=w, t=t
        )
        t_channel = repeat(self.t_channel_embed, 'c_g d -> b t c_g d', b=b, t=t)
        st_channel = repeat(self.st_channel_embed, 'c_g d -> b c_g d', b=b)
        sp_channel = repeat(
            self.sp_channel_embed, 'c_g d -> b h w c_g d', b=b, h=h, w=w
        )

        pos_embed_s_t = repeat(
            self.pos_embed[:t], 't d -> b h w t c_g d', b=b, h=h, w=w, c_g=s_t_c_g
        )
        m_embed_s_t = repeat(
            self.month_embed(months), 'b t d -> b h w t c_g d', h=h, w=w, c_g=s_t_c_g
        )

        pos_embed_t = repeat(self.pos_embed[:t], 't d -> b t c_g d', b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), 'b t d -> b t c_g d', c_g=t_c_g)
        t_zeros = torch.zeros(
            b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device
        )

        sp_zeros = torch.zeros(
            b, h, w, sp_c_g, sp_channel.shape[-1] * 2, device=sp_channel.device
        )

        st_zeros = torch.zeros(
            b, st_c_g, st_channel.shape[-1] * 3, device=st_channel.device
        )

        # find the resolution that each token represents, which will be
        # the number of pixels in a patch * the resolution of each pixel
        if patch_size is None:
            patch_size = self.base_patch_size
        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert h == w, (
            'get_2d_sincos_pos_embed_with_resolution currently requires that h==w'
        )
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25),
            h,
            torch.ones(b).to(s_t_x.device) * gsd_ratio,
            device=s_t_x.device,
        )
        spatial_embed = rearrange(spatial_embed, 'b (h w) d -> b h w d', h=h, w=w)
        spatial_embed_s_t = repeat(
            spatial_embed, 'b h w d -> b h w t c_g d', h=h, w=w, t=t, c_g=s_t_c_g
        )
        spatial_embed_s = repeat(
            spatial_embed, 'b h w d -> b h w c_g d', h=h, w=w, c_g=sp_c_g
        )

        s_t_embed = torch.cat(
            [s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1
        )
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)
        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed


class Encoder(GalileoBase):
    """Galileo encoder that projects and encodes multi-modal tokens."""

    cross_attn = False

    def __init__(
        self,
        max_patch_size: int = 8,
        embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        freeze_projections: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        """Initialize the Galileo encoder.

        Args:
            max_patch_size: Largest patch size used by flexible projections.
            embedding_size: Token embedding dimension.
            depth: Number of transformer blocks.
            mlp_ratio: MLP hidden-to-input ratio.
            num_heads: Number of attention heads.
            max_sequence_length: Maximum temporal sequence length.
            freeze_projections: Whether to freeze modality projection layers.
            drop_path: Stochastic depth probability.

        Returns:
            None.
        """
        super().__init__(
            embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            max_patch_size,
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
                    in_features=len(group), out_features=embedding_size
                )
                for group_name, group in self.time_groups.items()
            }
        )
        self.static_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(
                    in_features=len(group), out_features=embedding_size
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
        """Initialize linear layers with Xavier uniform weights.

        Args:
            m: Module instance to initialize.

        Returns:
            None.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        """Apply modality-specific projections to grouped input tensors.

        Args:
            s_t_x: Space-time inputs of shape ``[B, H, W, T, Cst_raw]``.
            sp_x: Space-only inputs of shape ``[B, H, W, Cs_raw]``.
            t_x: Time-only inputs of shape ``[B, T, Ct_raw]``.
            st_x: Static inputs of shape ``[B, Cstatic_raw]``.
            s_t_m: Space-time mask of shape ``[B, H, W, T, Cst_groups]``.
            sp_m: Space-only mask of shape ``[B, H, W, Cs_groups]``.
            t_m: Time-only mask of shape ``[B, T, Ct_groups]``.
            st_m: Static mask of shape ``[B, Cstatic_groups]``.
            patch_size: Patch size used for spatial tokenization.

        Returns:
            Tuple ``(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)`` where
            projected tensors have shapes ``[B, H', W', T, Cst_groups, D]``,
            ``[B, H', W', Cs_groups, D]``, ``[B, T, Ct_groups, D]``,
            ``[B, Cstatic_groups, D]`` and masks are downsampled/grouped to
            matching group axes.
        """
        b, h, w, t, _ = s_t_x.shape
        new_h, new_w = h // patch_size, w // patch_size

        s_t_l, sp_l, t_l, st_l, s_t_m_l, sp_m_l, t_m_l, st_m_l = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx, (channel_group, channel_idxs) in enumerate(
            self.space_time_groups.items()
        ):
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            if s_t_m_l[-1].min() == 0:
                s_t_l.append(
                    self.space_time_embed[channel_group](
                        s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                s_t_l.append(
                    torch.zeros(
                        b,
                        new_h,
                        new_w,
                        t,
                        self.embedding_size,
                        dtype=s_t_x.dtype,
                        device=s_t_x.device,
                    )
                )
        for idx, (channel_group, channel_idxs) in enumerate(self.space_groups.items()):
            sp_m_l.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            if sp_m_l[-1].min() == 0:
                sp_l.append(
                    self.space_embed[channel_group](
                        sp_x[:, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                sp_l.append(
                    torch.zeros(
                        b,
                        new_h,
                        new_w,
                        self.embedding_size,
                        dtype=sp_x.dtype,
                        device=sp_x.device,
                    )
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.time_groups.items()):
            t_m_l.append(t_m[:, :, idx])
            if t_m_l[-1].min() == 0:
                t_l.append(self.time_embed[channel_group](t_x[:, :, channel_idxs]))
            else:
                t_l.append(
                    torch.zeros(
                        b, t, self.embedding_size, dtype=t_x.dtype, device=t_x.device
                    )
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            st_m_l.append(st_m[:, idx])
            if st_m_l[-1].min() == 0:
                st_l.append(self.static_embed[channel_group](st_x[:, channel_idxs]))
            else:
                st_l.append(
                    torch.zeros(
                        b, self.embedding_size, dtype=st_x.dtype, device=st_x.device
                    )
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
        """Move unmasked tokens to front and trim masked padding.

        Args:
            x: Token tensor of shape ``[B, N, D]``.
            mask: Binary mask tensor of shape ``[B, N]`` where masked is non-zero.

        Returns:
            Tuple ``(x_trimmed, indices, mask_trimmed)`` with shapes
            ``[B, N', D]``, ``[B, N]``, and ``[B, N']``.
        """
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        # https://stackoverflow.com/a/68621610/2332296
        # move all non-masked values to the front of their rows
        sorted_mask, indices = torch.sort(
            (~mask).int(), dim=1, descending=True, stable=True
        )
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def add_removed_tokens(
        x: Tensor, indices: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Restore full token ordering after trimming masked tokens.

        Args:
            x: Trimmed token tensor of shape ``[B, N', D]``.
            indices: Original sorting indices of shape ``[B, N]``.
            mask: Trimmed mask tensor of shape ``[B, N']``.

        Returns:
            Tuple ``(x_full, mask_full)`` with shapes ``[B, N, D]`` and ``[B, N]``.
        """
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), 'd -> b t d', b=x.shape[0], t=indices.shape[1]
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
        # can't set value on leaf variable
        out = masked_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[~full_mask.bool()] = x[~mask.bool()]
        # then move them to their original positions
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
        input_res: int = BASE_GSD,
        exit_after: int | None = None,
        token_exit_cfg: dict[str, int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run encoder attention blocks over projected token groups.

        Args:
            s_t_x: Space-time tokens of shape ``[B, H, W, T, Cst, D]``.
            sp_x: Space-only tokens of shape ``[B, H, W, Cs, D]``.
            t_x: Time-only tokens of shape ``[B, T, Ct, D]``.
            st_x: Static tokens of shape ``[B, Cstatic, D]``.
            s_t_m: Space-time masks of shape ``[B, H, W, T, Cst]``.
            sp_m: Space-only masks of shape ``[B, H, W, Cs]``.
            t_m: Time-only masks of shape ``[B, T, Ct]``.
            st_m: Static masks of shape ``[B, Cstatic]``.
            months: Month indices of shape ``[B, T]``.
            patch_size: Patch size used for tokenization.
            input_res: Input pixel resolution in meters.
            exit_after: Optional number of blocks to execute.
            token_exit_cfg: Optional per-group early-exit block mapping.

        Returns:
            Tuple ``(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)`` with
            encoded token tensors and original grouped masks.
        """
        if token_exit_cfg is not None:
            exit_s_t, exit_sp, exit_t, exit_st = self.create_token_exit_ids(
                s_t_x, sp_x, t_x, st_x, token_exit_cfg
            )
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(
                exit_s_t, exit_sp, exit_t, exit_st, s_t_m, sp_m, t_m, st_m
            )
            # exited_tokens starts as linear projections!
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
        x, m = self.collapse_and_combine_hwtc(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )

        # we only care about the values >= 1 for this mask, since 2 just tells the decoder
        # to decode those tokens. From the perspective of the encoder, 1 and 2 are equivalent
        # since they both represent masked values
        new_m = m >= 1
        x, indices, new_m = self.remove_masked_tokens(
            x, new_m
        )  # new_m is shape (bsz, seq_len)

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, m >= 1)
            # still linear projections
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, m >= 1)

        for i_blk, blk in enumerate(self.blocks):
            if (exit_after is not None) and ((i_blk + 1) > exit_after):
                # if exit_after is N, then we exit after the Nth layer
                # if exit_after is 0, then all layers are skipped
                break

            # skip the 0th block since this is just the linear
            # projection
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                # half depth
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=x.detach(),
                    other=exited_tokens.detach(),
                )

            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            x = blk(x=x, y=None, attn_mask=~new_m.bool())

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            x = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=x.detach(),
                other=exited_tokens.detach(),
            )

        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
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
        """Average non-masked tokens into one embedding per sample.

        Args:
            s_t_x: Space-time tokens of shape ``[B, H, W, T, Cst, D]``.
            sp_x: Space-only tokens of shape ``[B, H, W, Cs, D]``.
            t_x: Time-only tokens of shape ``[B, T, Ct, D]``.
            st_x: Static tokens of shape ``[B, Cstatic, D]``.
            s_t_m: Space-time masks of shape ``[B, H, W, T, Cst]``.
            sp_m: Space-only masks of shape ``[B, H, W, Cs]``.
            t_m: Time-only masks of shape ``[B, T, Ct]``.
            st_m: Static masks of shape ``[B, Cstatic]``.

        Returns:
            Mean embedding of shape ``[B, D]``.
        """
        x, m = cls.collapse_and_combine_hwtc(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )
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
        """Average non-masked tokens independently per spatial patch.

        Args:
            s_t_x: Space-time tokens of shape ``[B, H, W, T, Cst, D]``.
            sp_x: Space-only tokens of shape ``[B, H, W, Cs, D]``.
            t_x: Time-only tokens of shape ``[B, T, Ct, D]``.
            st_x: Static tokens of shape ``[B, Cstatic, D]``.
            s_t_m: Space-time masks of shape ``[B, H, W, T, Cst]``.
            sp_m: Space-only masks of shape ``[B, H, W, Cs]``.
            t_m: Time-only masks of shape ``[B, T, Ct]``.
            st_m: Static masks of shape ``[B, Cstatic]``.

        Returns:
            Patch-wise mean embeddings of shape ``[B, H*W, D]``.
        """
        s_t_x = rearrange(s_t_x, 'b t_h t_w t c_g d -> b (t_h t_w) (t c_g) d')
        sp_x = rearrange(sp_x, 'b t_h t_w c_g d -> b (t_h t_w) c_g d')
        # repeat time tokens over space
        t_x = repeat(
            rearrange(t_x, 'b t c_g d -> b (t c_g) d'),
            'b n d -> b s n d',
            s=sp_x.shape[1],
        )
        st_x = repeat(st_x, 'b c_g d -> b s c_g d', s=sp_x.shape[1])
        s_t_m = rearrange(s_t_m, 'b t_h t_w t c_g-> b (t_h t_w) (t c_g)')
        sp_m = rearrange(sp_m, 'b t_h t_w c_g-> b (t_h t_w) c_g')
        t_m = repeat(
            rearrange(t_m, 'b t c_g -> b (t c_g)'), 'b n -> b s n', s=sp_x.shape[1]
        )
        st_m = repeat(st_m, 'b c_g -> b s c_g', s=sp_x.shape[1])

        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=2)  # B, S, N, D
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=2)  # B, S, N

        x_for_mean = x * (1 - m.unsqueeze(-1))

        return x_for_mean.sum(dim=2) / torch.sum(1 - m, -1, keepdim=True)

    def create_token_exit_ids(
        self,
        s_t_x: Tensor,
        sp_x: Tensor,
        t_x: Tensor,
        st_x: Tensor,
        token_exit_cfg: dict[str, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build per-token exit-depth tensors for early-exit execution.

        Args:
            s_t_x: Space-time tokens of shape ``[B, H, W, T, Cst, D]``.
            sp_x: Space-only tokens of shape ``[B, H, W, Cs, D]``.
            t_x: Time-only tokens of shape ``[B, T, Ct, D]``.
            st_x: Static tokens of shape ``[B, Cstatic, D]``.
            token_exit_cfg: Mapping from group name to encoder block exit index.

        Returns:
            Tuple of exit-id tensors with same shapes as corresponding inputs.
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
        space_time_x: torch.Tensor,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        static_x: torch.Tensor,
        space_time_mask: torch.Tensor,
        space_mask: torch.Tensor,
        time_mask: torch.Tensor,
        static_mask: torch.Tensor,
        months: torch.Tensor,
        patch_size: int,
        input_resolution_m: int = BASE_GSD,
        exit_after: int | None = None,
        token_exit_cfg: dict[str, int] | None = None,
        add_layernorm_on_exit: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode grouped multi-modal inputs.

        Args:
            space_time_x: Space-time inputs of shape ``[B, H, W, T, Cst_raw]``.
            space_x: Space-only inputs of shape ``[B, H, W, Cs_raw]``.
            time_x: Time-only inputs of shape ``[B, T, Ct_raw]``.
            static_x: Static inputs of shape ``[B, Cstatic_raw]``.
            space_time_mask: Space-time masks of shape ``[B, H, W, T, Cst_groups]``.
            space_mask: Space-only masks of shape ``[B, H, W, Cs_groups]``.
            time_mask: Time-only masks of shape ``[B, T, Ct_groups]``.
            static_mask: Static masks of shape ``[B, Cstatic_groups]``.
            months: Month indices of shape ``[B, T]``.
            patch_size: Spatial patch size for projections.
            input_resolution_m: Input pixel resolution in meters.
            exit_after: Optional number of attention blocks to execute.
            token_exit_cfg: Optional per-group early-exit configuration.
            add_layernorm_on_exit: Whether to apply final layer norm to outputs.

        Returns:
            Tuple ``(space_time_x, space_x, time_x, static_x, space_time_mask,``
            ``space_mask, time_mask, static_mask, months)``.
        """
        (
            space_time_x,
            space_x,
            time_x,
            static_x,
            space_time_mask,
            space_mask,
            time_mask,
            static_mask,
        ) = self.apply_linear_projection(
            space_time_x,
            space_x,
            time_x,
            static_x,
            space_time_mask,
            space_mask,
            time_mask,
            static_mask,
            patch_size,
        )

        if (exit_after is None) or (exit_after > 0):
            (
                space_time_x,
                space_x,
                time_x,
                static_x,
                space_time_mask,
                space_mask,
                time_mask,
                static_mask,
            ) = self.apply_attn(
                space_time_x,
                space_x,
                time_x,
                static_x,
                space_time_mask,
                space_mask,
                time_mask,
                static_mask,
                months,
                patch_size,
                input_resolution_m,
                exit_after=exit_after,
                token_exit_cfg=token_exit_cfg,
            )

        if add_layernorm_on_exit:
            space_time_x = self.norm(space_time_x)
            space_x = self.norm(space_x)
            time_x = self.norm(time_x)
            static_x = self.norm(static_x)
        return (
            space_time_x,
            space_x,
            time_x,
            static_x,
            space_time_mask,
            space_mask,
            time_mask,
            static_mask,
            months,
        )


class Galileo(Encoder):
    """Galileo encoder model.

    .. versionadded:: 0.9
    """


class Galileo_Weights(WeightsEnum):  # type: ignore[misc]
    """Galileo model weights.

    .. versionadded:: 0.9
    """

    GALILEO_NANO = Weights(
        url='https://hf.co/isaaccorley/galileo/resolve/09adbfeaffc50a4817578abca4c9e3a1723d571d/model_nano-ebaf045a.pth',
        transforms=nn.Identity(),
        meta={
            'publication': 'https://arxiv.org/abs/2502.09356',
            'repo': 'https://github.com/nasaharvest/galileo',
            'license': 'MIT',
            'encoder_config': {
                'embedding_size': 128,
                'depth': 4,
                'num_heads': 8,
                'mlp_ratio': 4,
                'max_sequence_length': 24,
                'freeze_projections': False,
                'drop_path': 0.1,
                'max_patch_size': 8,
            },
        },
    )
    GALILEO_TINY = Weights(
        url='https://hf.co/isaaccorley/galileo/resolve/09adbfeaffc50a4817578abca4c9e3a1723d571d/model_tiny-4f414eea.pth',
        transforms=nn.Identity(),
        meta={
            'publication': 'https://arxiv.org/abs/2502.09356',
            'repo': 'https://github.com/nasaharvest/galileo',
            'license': 'MIT',
            'encoder_config': {
                'embedding_size': 192,
                'depth': 12,
                'num_heads': 3,
                'mlp_ratio': 4,
                'max_sequence_length': 24,
                'freeze_projections': False,
                'drop_path': 0.1,
                'max_patch_size': 8,
            },
        },
    )
    GALILEO_BASE = Weights(
        url='https://hf.co/isaaccorley/galileo/resolve/09adbfeaffc50a4817578abca4c9e3a1723d571d/model_base-7f15d404.pth',
        transforms=nn.Identity(),
        meta={
            'publication': 'https://arxiv.org/abs/2502.09356',
            'repo': 'https://github.com/nasaharvest/galileo',
            'license': 'MIT',
            'encoder_config': {
                'embedding_size': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4,
                'max_sequence_length': 24,
                'freeze_projections': False,
                'drop_path': 0.1,
                'max_patch_size': 8,
            },
        },
    )


def galileo(
    weights: Galileo_Weights | None = None, *args: Any, **kwargs: Any
) -> Galileo:
    """Galileo model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2502.09356

    .. versionadded:: 0.9

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Galileo`.
        **kwargs: Additional keyword arguments to pass to :class:`Galileo`.

    Returns:
        A Galileo model.
    """
    if weights:
        model = Galileo(**weights.meta['encoder_config'])
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    else:
        model = Galileo(*args, **kwargs)

    return model
