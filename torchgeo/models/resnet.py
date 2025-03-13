# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ResNet models."""

from typing import Any

import kornia.augmentation as K
import timm
import torch
from timm.models import ResNet
from torchvision.models._api import Weights, WeightsEnum

from .swin import (
    _satlas_bands,
    _satlas_sentinel2_bands,
    _satlas_sentinel2_transforms,
    _satlas_transforms,
)
from .utils import load_pretrained

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA
_landsat_tm_toa_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA
_landsat_etm_toa_bands = [
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6_VCID_1',
    'B6_VCID_2',
    'B7',
    'B8',
]

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2
_landsat_etm_sr_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA
_landsat_oli_tirs_toa_bands = [
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6',
    'B7',
    'B8',
    'B9',
    'B10',
    'B11',
]

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
_landsat_oli_sr_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']

# https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
_sentinel2_toa_bands = [
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6',
    'B7',
    'B8',
    'B8a',
    'B9',
    'B10',
    'B11',
    'B12',
]

# https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
_sentinel2_rgb_bands = ['B4', 'B3', 'B2']

# https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
_sentinel1_bands = ['VV', 'VH']

# https://github.com/zhu-xlab/DeCUR/blob/f190e9a3895ef645c005c8c2fce287ffa5a937e3/src/transfer_classification_BE/linear_BE_resnet.py#L286
# Normalization by channel-wise band statistics
_mean_s1 = torch.tensor([-12.59, -20.26])
_std_s1 = torch.tensor([5.26, 5.91])
_ssl4eo_s12_transforms_s1 = K.AugmentationSequential(
    K.Resize(256),
    K.CenterCrop(224),
    K.Normalize(mean=_mean_s1, std=_std_s1),
    data_keys=None,
)

# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/linear_BE_moco.py#L167
# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/datasets/EuroSat/eurosat_dataset.py#L97
# Normalization either by 10K (for S2 uint16 input) or channel-wise with band statistics
_ssl4eo_s12_transforms_s2_10k = K.AugmentationSequential(
    K.Resize(256),
    K.CenterCrop(224),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(10000)),
    data_keys=None,
)

_mean_s2 = torch.tensor(
    [
        1612.9,
        1397.6,
        1322.3,
        1373.1,
        1561.0,
        2108.4,
        2390.7,
        2318.7,
        2581.0,
        837.7,
        22.0,
        2195.2,
        1537.4,
    ]
)
_std_s2 = torch.tensor(
    [
        791.0,
        854.3,
        878.7,
        1144.9,
        1127.5,
        1164.2,
        1276.0,
        1249.5,
        1345.9,
        577.5,
        47.5,
        1340.0,
        1142.9,
    ]
)
_ssl4eo_s12_transforms_s2_stats = K.AugmentationSequential(
    K.Resize(256),
    K.CenterCrop(224),
    K.Normalize(mean=_mean_s2, std=_std_s2),
    data_keys=None,
)

# Normalization only available for RGB dataset, defined here:
# https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/seco_dataset.py
_min = torch.tensor([3, 2, 0])
_max = torch.tensor([88, 103, 129])
_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
_seco_transforms = K.AugmentationSequential(
    K.Resize(256),
    K.CenterCrop(224),
    K.Normalize(mean=_min, std=_max - _min),
    K.Normalize(mean=torch.tensor(0), std=1 / torch.tensor(255)),
    K.Normalize(mean=_mean, std=_std),
    data_keys=None,
)

# Normalization only available for RGB dataset, defined here:
# https://github.com/sustainlab-group/geography-aware-ssl/blob/main/moco_fmow/main_moco_geo%2Btp.py#L287
_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
_gassl_transforms = K.AugmentationSequential(
    K.Resize(224),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    K.Normalize(mean=_mean, std=_std),
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


class ResNet18_Weights(WeightsEnum):  # type: ignore[misc]
    """ResNet-18 weights.

    For `timm <https://github.com/huggingface/pytorch-image-models>`_
    *resnet18* implementation.

    .. versionadded:: 0.4
    """

    LANDSAT_TM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_tm_toa_moco-1c691b4f.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_tm_toa_bands,
        },
    )

    LANDSAT_TM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_tm_toa_simclr-d2d38ace.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_tm_toa_bands,
        },
    )

    LANDSAT_ETM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_toa_moco-bb88689c.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_etm_toa_bands,
        },
    )

    LANDSAT_ETM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_toa_simclr-4d813f79.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_etm_toa_bands,
        },
    )

    LANDSAT_ETM_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_sr_moco-4f078acd.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_etm_sr_bands,
        },
    )

    LANDSAT_ETM_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_sr_simclr-8e8543b4.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_etm_sr_bands,
        },
    )

    LANDSAT_OLI_TIRS_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_tirs_toa_moco-a3002f51.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_oli_tirs_toa_bands,
        },
    )

    LANDSAT_OLI_TIRS_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_tirs_toa_simclr-b0635cc6.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_oli_tirs_toa_bands,
        },
    )

    LANDSAT_OLI_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_sr_moco-660e82ed.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_oli_sr_bands,
        },
    )

    LANDSAT_OLI_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_sr_simclr-7bced5be.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_oli_sr_bands,
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet18_sentinel2_all_moco/resolve/5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth',
        transforms=_ssl4eo_s12_transforms_s2_10k,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
            'bands': _sentinel2_toa_bands,
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet18_sentinel2_rgb_moco/resolve/e1c032e7785fd0625224cdb6699aa138bb304eec/resnet18_sentinel2_rgb_moco-e3a335e3.pth',
        transforms=_ssl4eo_s12_transforms_s2_10k,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 3,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
            'bands': _sentinel2_rgb_bands,
        },
    )

    SENTINEL2_RGB_SECO = Weights(
        url='https://hf.co/torchgeo/resnet18_sentinel2_rgb_seco/resolve/f8dcee692cf7142163b55a5c197d981fe0e717a0/resnet18_sentinel2_rgb_seco-cefca942.pth',
        transforms=_seco_transforms,
        meta={
            'dataset': 'SeCo Dataset',
            'in_chans': 3,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2103.16607',
            'repo': 'https://github.com/ServiceNow/seasonal-contrast',
            'ssl_method': 'seco',
            'bands': _sentinel2_rgb_bands,
        },
    )


class ResNet50_Weights(WeightsEnum):  # type: ignore[misc]
    """ResNet-50 weights.

    For `timm <https://github.com/huggingface/pytorch-image-models>`_
    *resnet50* implementation.

    .. versionadded:: 0.4
    """

    FMOW_RGB_GASSL = Weights(
        url='https://hf.co/torchgeo/resnet50_fmow_rgb_gassl/resolve/fe8a91026cf9104f1e884316b8e8772d7af9052c/resnet50_fmow_rgb_gassl-da43d987.pth',
        transforms=_gassl_transforms,
        meta={
            'dataset': 'fMoW Dataset',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2011.09980',
            'repo': 'https://github.com/sustainlab-group/geography-aware-ssl',
            'ssl_method': 'gassl',
        },
    )

    LANDSAT_TM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_tm_toa_moco-ba1ce753.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_tm_toa_bands,
        },
    )

    LANDSAT_TM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_tm_toa_simclr-a1c93432.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_tm_toa_bands,
        },
    )

    LANDSAT_ETM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_toa_moco-e9a84d5a.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_etm_toa_bands,
        },
    )

    LANDSAT_ETM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_toa_simclr-70b5575f.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_etm_toa_bands,
        },
    )

    LANDSAT_ETM_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_sr_moco-1266cde3.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_etm_sr_bands,
        },
    )

    LANDSAT_ETM_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_sr_simclr-e5d185d7.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_etm_sr_bands,
        },
    )

    LANDSAT_OLI_TIRS_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_tirs_toa_moco-de7f5e0f.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_oli_tirs_toa_bands,
        },
    )

    LANDSAT_OLI_TIRS_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_tirs_toa_simclr-030cebfe.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_oli_tirs_toa_bands,
        },
    )

    LANDSAT_OLI_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_sr_moco-ff580dad.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
            'bands': _landsat_oli_sr_bands,
        },
    )

    LANDSAT_OLI_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_sr_simclr-94f78913.pth',
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
            'bands': _landsat_oli_sr_bands,
        },
    )

    SENTINEL1_ALL_DECUR = Weights(
        url='https://huggingface.co/torchgeo/decur/resolve/9328eeb90c686a88b30f8526ed757b4bc0f12027/rn50_ssl4eo-s12_sar_decur_ep100-f0e69ba2.pth',
        transforms=_ssl4eo_s12_transforms_s1,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 2,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2309.05300',
            'repo': 'https://github.com/zhu-xlab/DeCUR',
            'ssl_method': 'decur',
            'bands': _sentinel1_bands,
        },
    )

    SENTINEL1_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel1_all_moco/resolve/e79862c667853c10a709bdd77ea8ffbad0e0f1cf/resnet50_sentinel1_all_moco-906e4356.pth',
        transforms=_ssl4eo_s12_transforms_s1,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 2,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
            'bands': _sentinel1_bands,
        },
    )

    SENTINEL2_ALL_DECUR = Weights(
        url='https://huggingface.co/torchgeo/decur/resolve/eba7ae5945d482a4319be046d34b552db5dd9950/rn50_ssl4eo-s12_ms_decur_ep100-fc6b09ff.pth',
        transforms=_ssl4eo_s12_transforms_s2_10k,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2309.05300',
            'repo': 'https://github.com/zhu-xlab/DeCUR',
            'ssl_method': 'decur',
            'bands': _sentinel2_toa_bands,
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_all_dino/resolve/d7f14bf5530d70ac69d763e58e77e44dbecfec7c/resnet50_sentinel2_all_dino-d6c330e9.pth',
        transforms=_ssl4eo_s12_transforms_s2_10k,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'dino',
            'bands': _sentinel2_toa_bands,
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth',
        transforms=_ssl4eo_s12_transforms_s2_10k,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
            'bands': _sentinel2_toa_bands,
        },
    )

    SENTINEL2_MI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet50_mi_ms-da5413d2.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_MI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet50_mi_rgb-e79bb7fe.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_bands,
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_rgb_moco/resolve/efd9723b59a88e9dc1420dc1e96afb25b0630a3c/resnet50_sentinel2_rgb_moco-2b57ba8b.pth',
        transforms=_ssl4eo_s12_transforms_s2_10k,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
            'bands': _sentinel2_rgb_bands,
        },
    )

    SENTINEL2_RGB_SECO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_rgb_seco/resolve/fbd07b02a8edb8fc1035f7957160deed4321c145/resnet50_sentinel2_rgb_seco-018bf397.pth',
        transforms=_seco_transforms,
        meta={
            'dataset': 'SeCo Dataset',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2103.16607',
            'repo': 'https://github.com/ServiceNow/seasonal-contrast',
            'ssl_method': 'seco',
            'bands': _sentinel2_rgb_bands,
        },
    )

    SENTINEL2_SI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet50_si_ms-1f454cc6.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_SI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet50_si_rgb-45fc6972.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_bands,
        },
    )


class ResNet152_Weights(WeightsEnum):  # type: ignore[misc]
    """ResNet-152 weights.

    For `timm <https://github.com/huggingface/pytorch-image-models>`_
    *resnet152* implementation.

    .. versionadded:: 0.6
    """

    SENTINEL2_MI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet152_mi_ms-fd35b4bb.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_MI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet152_mi_rgb-67563ac5.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_bands,
        },
    )

    SENTINEL2_SI_MS_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet152_si_ms-4500c6cb.pth',
        transforms=_satlas_sentinel2_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_sentinel2_bands,
        },
    )

    SENTINEL2_SI_RGB_SATLAS = Weights(
        url='https://hf.co/torchgeo/satlas/resolve/081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet152_si_rgb-f4d24c3c.pth',
        transforms=_satlas_transforms,
        meta={
            'dataset': 'SatlasPretrain',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.15660',
            'repo': 'https://github.com/allenai/satlaspretrain_models',
            'bands': _satlas_bands,
        },
    )


def resnet18(
    weights: ResNet18_Weights | None = None, *args: Any, **kwargs: Any
) -> ResNet:
    """ResNet-18 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385

    .. versionadded:: 0.4

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`
        **kwargs: Additional keyword arguments to pass to :func:`timm.create_model`

    Returns:
        A ResNet-18 model.
    """
    if weights:
        kwargs['in_chans'] = weights.meta['in_chans']

    model: ResNet = timm.create_model('resnet18', *args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= {'fc.weight', 'fc.bias'}
        assert not unexpected_keys

    return model


def resnet50(
    weights: ResNet50_Weights | None = None, *args: Any, **kwargs: Any
) -> ResNet:
    """ResNet-50 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385

    .. versionchanged:: 0.4
       Switched to multi-weight support API.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keyword arguments to pass to :func:`timm.create_model`.

    Returns:
        A ResNet-50 model.
    """
    model: ResNet = timm.create_model('resnet50', *args, **kwargs)

    if weights:
        pretrained_cfg = {}
        pretrained_cfg['first_conv'] = 'conv1'
        in_chans = kwargs.get('in_chans', 3)
        missing_keys, unexpected_keys = load_pretrained(
            model,
            weights=weights,
            pretrained_cfg=pretrained_cfg,
            in_chans=in_chans,
            strict=False,
        )
        assert set(missing_keys) <= {'fc.weight', 'fc.bias'}
        assert not unexpected_keys

    return model


def resnet152(
    weights: ResNet152_Weights | None = None, *args: Any, **kwargs: Any
) -> ResNet:
    """ResNet-152 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keyword arguments to pass to :func:`timm.create_model`.

    Returns:
        A ResNet-152 model.
    """
    if weights:
        kwargs['in_chans'] = weights.meta['in_chans']

    model: ResNet = timm.create_model('resnet152', *args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= {'fc.weight', 'fc.bias'}
        assert not unexpected_keys

    return model
