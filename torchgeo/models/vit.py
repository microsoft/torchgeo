# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Vision Transformer models."""

from typing import Any, cast

import kornia.augmentation as K
import timm
import torch
from timm.models._features import FeatureGetterNet
from timm.models.vision_transformer import VisionTransformer
from torchvision.models._api import Weights, WeightsEnum

from .resnet import (
    _landsat_etm_sr_bands,
    _landsat_etm_toa_bands,
    _landsat_oli_sr_bands,
    _landsat_oli_tirs_toa_bands,
    _landsat_tm_toa_bands,
    _sentinel2_toa_bands,
)

# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/linear_BE_moco.py#L167
# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/datasets/EuroSat/eurosat_dataset.py#L97
# Normalization either by 10K or channel-wise with band statistics
_zhu_xlab_transforms = K.AugmentationSequential(
    K.Resize(256),
    K.CenterCrop(224),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(10000)),
    data_keys=None,
)

# https://github.com/microsoft/torchgeo/blob/8b53304d42c269f9001cb4e861a126dc4b462606/torchgeo/datamodules/ssl4eo_benchmark.py#L43
_ssl4eo_l_transforms = K.AugmentationSequential(
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    K.CenterCrop((224, 224)),
    data_keys=None,
)

# https://github.com/pytorch/vision/pull/6883
# https://github.com/pytorch/vision/pull/7107
# Can be removed once torchvision>=0.15 is required
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]


class ViTSmall16_Weights(WeightsEnum):  # type: ignore[misc]
    """Vision Transformer Small Patch Size 16 weights.

    For `timm <https://github.com/huggingface/pytorch-image-models>`_
    *vit_small_patch16_224* implementation.

    .. versionadded:: 0.4
    """

    LANDSAT_TM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_tm_toa_moco-a1c967d8.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_tm_toa_bands,
        },
    )

    LANDSAT_TM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_tm_toa_simclr-7c2d9799.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_tm_toa_bands,
        },
    )

    LANDSAT_ETM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_etm_toa_moco-26d19bcf.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_etm_toa_bands,
        },
    )

    LANDSAT_ETM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_etm_toa_simclr-34fb12cb.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_etm_toa_bands,
        },
    )

    LANDSAT_ETM_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_etm_sr_moco-eaa4674e.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_etm_sr_bands,
        },
    )

    LANDSAT_ETM_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_etm_sr_simclr-a14c466a.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_etm_sr_bands,
        },
    )

    LANDSAT_OLI_TIRS_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_oli_tirs_toa_moco-c7c2cceb.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_oli_tirs_toa_bands,
        },
    )

    LANDSAT_OLI_TIRS_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_oli_tirs_toa_simclr-ad43e9a4.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_oli_tirs_toa_bands,
        },
    )

    LANDSAT_OLI_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_oli_sr_moco-c9b8898d.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_oli_sr_bands,
        },
    )

    LANDSAT_OLI_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/vits16_landsat_oli_sr_simclr-4e8f6102.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_oli_sr_bands,
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url='https://hf.co/torchgeo/vit_small_patch16_224_sentinel2_all_dino/resolve/5b41dd418a79de47ac9f5be3e035405a83818a62/vit_small_patch16_224_sentinel2_all_dino-36bcc127.pth',
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'dino',
            'bands': _sentinel2_toa_bands,
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/vit_small_patch16_224_sentinel2_all_moco/resolve/1cb683f6c14739634cdfaaceb076529adf898c74/vit_small_patch16_224_sentinel2_all_moco-67c9032d.pth',
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'vit_small_patch16_224',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
            'bands': _sentinel2_toa_bands,
        },
    )


def vit_small_patch16_224(
    weights: ViTSmall16_Weights | None = None, *args: Any, **kwargs: Any
) -> VisionTransformer | FeatureGetterNet:
    """Vision Transform (ViT) small patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2010.11929

    .. versionadded:: 0.4

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keyword arguments to pass to :func:`timm.create_model`.

    Returns:
        A ViT small 16 model.
    """
    if weights:
        kwargs['in_chans'] = weights.meta['in_chans']

    model: VisionTransformer | FeatureGetterNet = timm.create_model(
        'vit_small_patch16_224', *args, **kwargs
    )

    if 'features_only' in kwargs:
        model = cast(FeatureGetterNet, model)
        target_model: VisionTransformer = cast(VisionTransformer, model.model)
    else:
        model = cast(VisionTransformer, model)
        target_model = model

    if weights:
        missing_keys, unexpected_keys = target_model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= {'head.weight', 'head.bias'}
        # used when features_only = True
        assert set(unexpected_keys) <= {'norm.weight', 'norm.bias'}

    return model
