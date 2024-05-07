# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ResNet models."""

from typing import Any

import kornia.augmentation as K
import timm
import torch
from timm.models import ResNet
from torchvision.models._api import Weights, WeightsEnum

# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/linear_BE_moco.py#L167  # noqa: E501
# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/datasets/EuroSat/eurosat_dataset.py#L97  # noqa: E501
# Normalization either by 10K or channel-wise with band statistics
_zhu_xlab_transforms = K.AugmentationSequential(
    K.Resize(256),
    K.CenterCrop(224),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(10000)),
    data_keys=None,
)

# Normalization only available for RGB dataset, defined here:
# https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/seco_dataset.py  # noqa: E501
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
# https://github.com/sustainlab-group/geography-aware-ssl/blob/main/moco_fmow/main_moco_geo%2Btp.py#L287  # noqa: E501
_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
_gassl_transforms = K.AugmentationSequential(
    K.Resize(224),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    K.Normalize(mean=_mean, std=_std),
    data_keys=None,
)

# https://github.com/microsoft/torchgeo/blob/8b53304d42c269f9001cb4e861a126dc4b462606/torchgeo/datamodules/ssl4eo_benchmark.py#L43  # noqa: E501
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
    """ResNet18 weights.

    For `timm <https://github.com/rwightman/pytorch-image-models>`_
    *resnet18* implementation.

    .. versionadded:: 0.4
    """

    LANDSAT_TM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_tm_toa_moco-1c691b4f.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_TM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_tm_toa_simclr-d2d38ace.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_ETM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_toa_moco-bb88689c.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_ETM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_toa_simclr-4d813f79.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_ETM_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_sr_moco-4f078acd.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_ETM_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_etm_sr_simclr-8e8543b4.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_OLI_TIRS_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_tirs_toa_moco-a3002f51.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_OLI_TIRS_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_tirs_toa_simclr-b0635cc6.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_OLI_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_sr_moco-660e82ed.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_OLI_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet18_landsat_oli_sr_simclr-7bced5be.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet18_sentinel2_all_moco/resolve/5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth',  # noqa: E501
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet18_sentinel2_rgb_moco/resolve/e1c032e7785fd0625224cdb6699aa138bb304eec/resnet18_sentinel2_rgb_moco-e3a335e3.pth',  # noqa: E501
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 3,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
        },
    )

    SENTINEL2_RGB_SECO = Weights(
        url='https://hf.co/torchgeo/resnet18_sentinel2_rgb_seco/resolve/f8dcee692cf7142163b55a5c197d981fe0e717a0/resnet18_sentinel2_rgb_seco-cefca942.pth',  # noqa: E501
        transforms=_seco_transforms,
        meta={
            'dataset': 'SeCo Dataset',
            'in_chans': 3,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2103.16607',
            'repo': 'https://github.com/ServiceNow/seasonal-contrast',
            'ssl_method': 'seco',
        },
    )


class ResNet50_Weights(WeightsEnum):  # type: ignore[misc]
    """ResNet50 weights.

    For `timm <https://github.com/rwightman/pytorch-image-models>`_
    *resnet50* implementation.

    .. versionadded:: 0.4
    """

    FMOW_RGB_GASSL = Weights(
        url='https://hf.co/torchgeo/resnet50_fmow_rgb_gassl/resolve/fe8a91026cf9104f1e884316b8e8772d7af9052c/resnet50_fmow_rgb_gassl-da43d987.pth',  # noqa: E501
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
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_tm_toa_moco-ba1ce753.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_TM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_tm_toa_simclr-a1c93432.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_ETM_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_toa_moco-e9a84d5a.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_ETM_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_toa_simclr-70b5575f.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 9,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_ETM_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_sr_moco-1266cde3.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_ETM_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_etm_sr_simclr-e5d185d7.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 6,
            'model': 'resnet18',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_OLI_TIRS_TOA_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_tirs_toa_moco-de7f5e0f.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_OLI_TIRS_TOA_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_tirs_toa_simclr-030cebfe.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 11,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    LANDSAT_OLI_SR_MOCO = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_sr_moco-ff580dad.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'moco',
        },
    )

    LANDSAT_OLI_SR_SIMCLR = Weights(
        url='https://hf.co/torchgeo/ssl4eo_landsat/resolve/1c88bb51b6e17a21dde5230738fa38b74bd74f76/resnet50_landsat_oli_sr_simclr-94f78913.pth',  # noqa: E501
        transforms=_ssl4eo_l_transforms,
        meta={
            'dataset': 'SSL4EO-L',
            'in_chans': 7,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2306.09424',
            'repo': 'https://github.com/microsoft/torchgeo',
            'ssl_method': 'simclr',
        },
    )

    SENTINEL1_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel1_all_moco/resolve/e79862c667853c10a709bdd77ea8ffbad0e0f1cf/resnet50_sentinel1_all_moco-906e4356.pth',  # noqa: E501
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 2,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_all_dino/resolve/d7f14bf5530d70ac69d763e58e77e44dbecfec7c/resnet50_sentinel2_all_dino-d6c330e9.pth',  # noqa: E501
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'dino',
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth',  # noqa: E501
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 13,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_rgb_moco/resolve/efd9723b59a88e9dc1420dc1e96afb25b0630a3c/resnet50_sentinel2_rgb_moco-2b57ba8b.pth',  # noqa: E501
        transforms=_zhu_xlab_transforms,
        meta={
            'dataset': 'SSL4EO-S12',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2211.07044',
            'repo': 'https://github.com/zhu-xlab/SSL4EO-S12',
            'ssl_method': 'moco',
        },
    )

    SENTINEL2_RGB_SECO = Weights(
        url='https://hf.co/torchgeo/resnet50_sentinel2_rgb_seco/resolve/fbd07b02a8edb8fc1035f7957160deed4321c145/resnet50_sentinel2_rgb_seco-018bf397.pth',  # noqa: E501
        transforms=_seco_transforms,
        meta={
            'dataset': 'SeCo Dataset',
            'in_chans': 3,
            'model': 'resnet50',
            'publication': 'https://arxiv.org/abs/2103.16607',
            'repo': 'https://github.com/ServiceNow/seasonal-contrast',
            'ssl_method': 'seco',
        },
    )


def resnet18(
    weights: ResNet18_Weights | None = None, *args: Any, **kwargs: Any
) -> ResNet:
    """ResNet-18 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    .. versionadded:: 0.4

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`
        **kwargs: Additional keywork arguments to pass to :func:`timm.create_model`

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

    * https://arxiv.org/pdf/1512.03385.pdf

    .. versionchanged:: 0.4
       Switched to multi-weight support API.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keywork arguments to pass to :func:`timm.create_model`.

    Returns:
        A ResNet-50 model.
    """
    if weights:
        kwargs['in_chans'] = weights.meta['in_chans']

    model: ResNet = timm.create_model('resnet50', *args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= {'fc.weight', 'fc.bias'}
        assert not unexpected_keys

    return model
