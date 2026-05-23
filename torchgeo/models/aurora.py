# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Aurora models."""

from typing import Any, cast

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

# Aurora operates on the raw unnormalized data.
_aurora_transforms = nn.Identity()

_aurora_meta = {
    'dataset': 'Aurora',
    'model': None,
    'resolution': None,
    'patch_size': None,
    'surf_vars': None,
    'atmos_vars': None,
    'static_vars': None,
    'architecture': '3D Swin Transformer U-Net',
    'encoder': '3D Perceiver',
    'hf_repo': 'microsoft/aurora',
    'filename': None,
    'publication': 'https://arxiv.org/abs/2409.16252',
    'repo': 'https://github.com/microsoft/aurora',
    'license': 'MIT',
    'lead-time': '6 hours',
    'units': 'degrees',
}


class Aurora_Weights(WeightsEnum):
    """Aurora weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.13063

    .. versionadded:: 0.8
    """

    HRES_T0_PRETRAINED_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-pretrained.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-pretrained.ckpt',
            'model': 'AuroraPretrained',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_PRETRAINED_12HR_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-12h-pretrained.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-12h-pretrained.ckpt',
            'model': 'Aurora12hPretrained',
            'lead-time': '12 hours',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_PRETRAINED_SMALL_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-small-pretrained.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-small-pretrained.ckpt',
            'model': 'AuroraSmallPretrained',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-finetuned.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-finetuned.ckpt',
            'model': 'Aurora',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_T0_HIGH_RES_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.1-finetuned.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.1-finetuned.ckpt',
            'model': 'AuroraHighRes',
            'resolution': 0.1,
            'patch_size': 10,
            'surf_vars': ('2t', '10u', '10v', 'msl'),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt'),
        },
    )

    HRES_CAMS_AIR_POLLUTION_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.4-air-pollution.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.4-air-pollution.ckpt',
            'model': 'AuroraAirPollution',
            'resolution': 0.4,
            'patch_size': 3,
            'surf_vars': (
                '2t',
                '10u',
                '10v',
                'msl',
                'pm1',
                'pm2p5',
                'pm10',
                'tcco',
                'tc_no',
                'tcno2',
                'gtco3',
                'tcso2',
            ),
            'atmos_vars': ('z', 'u', 'v', 't', 'q', 'co', 'no', 'no2', 'go3', 'so2'),
            'static_vars': (
                'lsm',
                'z',
                'slt',
                'static_ammonia',
                'static_ammonia_log',
                'static_co',
                'static_co_log',
                'static_nox',
                'static_nox_log',
                'static_so2',
                'static_so2_log',
            ),
        },
    )

    HRES_WAM0_WAVE_AURORA = Weights(
        url='https://huggingface.co/microsoft/aurora/resolve/74598e8c65d53a96077c08bb91acdfa5525340c9/aurora-0.25-wave.ckpt',
        transforms=_aurora_transforms,
        meta=_aurora_meta
        | {
            'filename': 'aurora-0.25-wave.ckpt',
            'model': 'AuroraWave',
            'resolution': 0.25,
            'patch_size': 4,
            'surf_vars': (
                '2t',
                '10u',
                '10v',
                'msl',
                'swh',
                'mwd',
                'mwp',
                'pp1d',
                'shww',
                'mdww',
                'mpww',
                'shts',
                'mdts',
                'mpts',
                'swh1',
                'mwd1',
                'mwp1',
                'swh2',
                'mwd2',
                'mwp2',
                '10u_wave',
                '10v_wave',
                'wind',
            ),
            'atmos_vars': ('z', 'u', 'v', 't', 'q'),
            'static_vars': ('lsm', 'z', 'slt', 'wmb', 'lat_mask'),
        },
    )


def aurora_swin_unet(
    weights: Aurora_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """Aurora model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.13063

    This dataset requires the following additional library to be installed:

    * `microsoft-aurora <https://pypi.org/project/microsoft-aurora/>`_ to load the models.

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to ``aurora.Aurora``
        **kwargs: Additional keyword arguments to pass to ``aurora.Aurora``

    Returns:
        An Aurora model.
    """
    aurora = lazy_import('aurora')

    if weights is None:
        model = aurora.Aurora(*args, **kwargs)
    else:
        model = getattr(aurora, weights.meta['model'])(*args, **kwargs)
        model.load_checkpoint(
            repo=weights.meta['hf_repo'], name=weights.meta['filename']
        )

    return cast(nn.Module, model)
