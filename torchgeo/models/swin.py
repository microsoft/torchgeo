# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Swin Transformer models."""

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
_satlas_transforms = nn.Sequential(
    T.CenterCrop(256), T.Normalize(mean=[0], std=[255], inplace=True)
)

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
    T.CenterCrop(256), T.Normalize(mean=_mean, std=_std, inplace=True)
)

_satlas_landsat_bands = tuple(f'B{i:02}' for i in range(1, 12))
_satlas_landsat_transforms = nn.Sequential(
    T.CenterCrop(256), T.Normalize(mean=[4000], std=[16320], inplace=True)
)


class Swin_V2_T_Weights(WeightsEnum):
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


class Swin_V2_B_Weights(WeightsEnum):
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


class SwinBackbone_Weights(WeightsEnum):
    """SwinBackbone weights parent class.

    These weights contain the encoder weights and optionally the backbone layernorm
    weights. To select whether layernorm weights are returned pass `include_norms`
    to get_state_dict (default is false).

    .. versionadded:: 0.8
    """

    def get_state_dict(
        self, include_norms: bool = False, *args: Any, **kwargs: Any
    ) -> Any:
        """Get the state dict for this model from provided url, optionally including backbone layernorm weights.

        Args:
            include_norms (bool): Whether to also return backbone layernorm weights.
            *args (Any): anything passed to WeightsEnum get_state_dict.
            **kwargs (Any): anything passed to WeightsEnum get_state_dict.

        Returns:
            dict with state dict only if include_norms is False,
            dict with 'state_dict' and 'feat_norms_state_dict' if include_norms is True.
        """
        full_state_dict = WeightsEnum.get_state_dict(self, *args, **kwargs)
        if include_norms:
            return full_state_dict
        else:
            return full_state_dict['state_dict']


class Swin_T_Weights(SwinBackbone_Weights):
    """Swin Transformer Tiny weights.

    For `torchvision <https://github.com/pytorch/vision>`_
    *swin_t* implementation.

    .. versionadded:: 0.8
    """

    CITYSCAPES_SEMSEG = Weights(
        url='https://hf.co/blaz-r/swin_tiny_cityscapes_semantic_torchvision/resolve/0fc235be19c60ae5873ee0e569561c4e43f403ba/swin_tiny_cityscapes_semantic.pth',
        transforms=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        meta={
            'dataset': 'Cityscapes - semantic segmentation',
            'in_chans': 3,
            'model': 'SwinTransformer Tiny',
            'publication': 'https://arxiv.org/abs/2112.01527',
            'repo': 'https://github.com/facebookresearch/Mask2Former/',
            'window_size': 7,
        },
    )


class Swin_S_Weights(SwinBackbone_Weights):
    """Swin Transformer Small weights.

    For `torchvision <https://github.com/pytorch/vision>`_
    *swin_s* implementation.

    .. versionadded:: 0.8
    """

    CITYSCAPES_SEMSEG = Weights(
        url='https://hf.co/blaz-r/swin_small_cityscapes_semantic_torchvision/resolve/97ea7dddaa2f62b3b5de85e16e2597f1635598d3/swin_small_cityscapes_semantic.pth',
        transforms=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        meta={
            'dataset': 'Cityscapes - semantic segmentation',
            'in_chans': 3,
            'model': 'SwinTransformer Small',
            'publication': 'https://arxiv.org/abs/2112.01527',
            'repo': 'https://github.com/facebookresearch/Mask2Former/',
            'window_size': 7,
        },
    )


class Swin_B_Weights(SwinBackbone_Weights):
    """Swin Transformer Base weights.

    For `torchvision <https://github.com/pytorch/vision>`_
    *swin_b* implementation.

    .. versionadded:: 0.8
    """

    CITYSCAPES_SEMSEG = Weights(
        url='https://hf.co/blaz-r/swin_base_cityscapes_semantic_torchvision/resolve/972003c5f18caaa5fc07f9db74ba2dc69eb6c051/swin_base_cityscapes_semantic.pth',
        transforms=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        meta={
            'dataset': 'Cityscapes - semantic segmentation',
            'in_chans': 3,
            'model': 'SwinTransformer Base',
            'publication': 'https://arxiv.org/abs/2112.01527',
            'repo': 'https://github.com/facebookresearch/Mask2Former/',
            'window_size': 12,
        },
    )


def swin_t(
    weights: Swin_T_Weights | None = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer tiny model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2103.14030

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keyword arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Tiny model.
    """
    model: SwinTransformer = torchvision.models.swin_t(weights=None, *args, **kwargs)

    if weights:
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        # same as for swinv2
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # some weights do not contain final norm and cls head weights
        assert set(missing_keys) <= {
            'norm.weight',
            'norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model


def swin_s(
    weights: Swin_S_Weights | None = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer small model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2103.14030

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keyword arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Small model.
    """
    model: SwinTransformer = torchvision.models.swin_s(weights=None, *args, **kwargs)

    if weights:
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        # same as for swinv2
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # some weights do not contain final norm and cls head weights
        assert set(missing_keys) <= {
            'norm.weight',
            'norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model


def swin_b(
    weights: Swin_B_Weights | None = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer base model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2103.14030

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keyword arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Base model.
    """
    if weights:
        # here we use the class directly to support non-default larger window-size
        window_size = weights.meta.get('window_size', 7)
        kwargs |= {
            'patch_size': [4, 4],
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': [window_size, window_size],
            'stochastic_depth_prob': 0.5,
        }
        model: SwinTransformer = torchvision.models.SwinTransformer(*args, **kwargs)

        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels  # type: ignore[not-subscriptable]
        # same as for swinv2
        model.features[0][0] = torch.nn.Conv2d(  # type: ignore[invalid-assignment]
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # some weights do not contain final norm and cls head weights
        assert set(missing_keys) <= {
            'norm.weight',
            'norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys
    else:
        model = torchvision.models.swin_b(weights=None, *args, **kwargs)

    return model


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
