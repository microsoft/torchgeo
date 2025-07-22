# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Swin v2 Transformer models."""

from typing import Any

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from torchvision.models import SwinTransformer
from torchvision.models._api import Weights, WeightsEnum

# All Satlas transforms include:
# https://github.com/allenai/satlas/blob/main/satlas/cmd/model/train.py#L49
#
# Information about sensor-specific normalization can be found at:
# https://github.com/allenai/satlas/blob/main/Normalization.md
_satlas_bands = ('B04', 'B03', 'B02')
_satlas_transforms = nn.Sequential(T.CenterCrop(256), T.Normalize(mean=[0], std=[255]))

_satlas_sentinel2_bands = (
    'B04',
    'B03',
    'B02',
    'B05',
    'B06',
    'B07',
    'B08',
    'B11',
    'B12',
)
_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_std = [255, 255, 255, 8160, 8160, 8160, 8160, 8160, 8160]
_satlas_sentinel2_transforms = nn.Sequential(
    T.CenterCrop(256),
    T.Normalize(mean=_mean, std=_std),
    T.Lambda(lambda x: x.clip(0, 1)),
)

_satlas_landsat_bands = tuple(f'B{i:02}' for i in range(1, 12))
_satlas_landsat_transforms = nn.Sequential(
    T.CenterCrop(256),
    T.Normalize(mean=[4000], std=[16320]),
    T.Lambda(lambda x: x.clip(0, 1)),
)


class Swin_V2_T_Weights(WeightsEnum):  # type: ignore[misc]
    """Swin Transformer v2 Tiny weights.

    For `torchvision <https://github.com/pytorch/vision>`_
    *swin_v2_t* implementation.

    .. versionadded:: 0.6
    """

    SENTINEL2_MI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swint_mi_ms-d8c659e3.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'swin_v2_t',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_MI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swint_mi_rgb-424d91f4.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'swin_v2_t',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_bands,
        },
    )

    SENTINEL2_SI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swint_si_ms-bc68e396.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'swin_v2_t',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_SI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swint_si_rgb-0c1a96e0.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'swin_v2_t',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_bands,
        },
    )


class Swin_V2_B_Weights(WeightsEnum):  # type: ignore[misc]
    """Swin Transformer v2 Base weights.

    For `torchvision <https://github.com/pytorch/vision>`_
    *swin_v2_b* implementation.

    .. versionadded:: 0.6
    """

    NAIP_RGB_MI_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/aerial_swinb_mi-326d69e1.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': ('R', 'G', 'B'),
        },
    )

    NAIP_RGB_SI_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/aerial_swinb_si-e4169eb1.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': ('R', 'G', 'B'),
        },
    )

    LANDSAT_MI_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/landsat_swinb_mi-6b4a1cda.pth',
        transforms=_satlas_landsat_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 11,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_landsat_bands,
        },
    )

    LANDSAT_SI_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/landsat_swinb_si-4af978f6.pth',
        transforms=_satlas_landsat_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 11,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_landsat_bands,
        },
    )

    SENTINEL1_MI_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel1_swinb_mi-f6c43d97.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 2,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': ('VH', 'VV'),
        },
    )

    SENTINEL1_SI_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel1_swinb_si-3981c153.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 2,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': ('VH', 'VV'),
        },
    )

    SENTINEL2_MI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swinb_mi_ms-39c86721.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_MI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swinb_mi_rgb-4efa210c.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_bands,
        },
    )

    SENTINEL2_SI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swinb_si_ms-fe22a12c.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_SI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_swinb_si_rgb-156a98d5.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'swin_v2_b',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlas',
            'bands': _satlas_bands,
        },
    )


def swin_v2_t(
    weights: Swin_V2_T_Weights | None = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer v2 tiny model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2111.09883

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keyword arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Tiny model.
    """
    model: SwinTransformer = torchvision.models.swin_v2_t(weights=None, *args, **kwargs)

    if weights:
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        # https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/backbones.py#L27
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= set()
        assert not unexpected_keys

    return model


def swin_v2_b(
    weights: Swin_V2_B_Weights | None = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer v2 base model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2111.09883

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keyword arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Base model.
    """
    model: SwinTransformer = torchvision.models.swin_v2_b(weights=None, *args, **kwargs)

    if weights:
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        # https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/backbones.py#L27
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= set()
        assert not unexpected_keys

    return model
