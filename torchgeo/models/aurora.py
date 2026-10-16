# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Aurora models."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from torch import nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import Sample, lazy_import

if TYPE_CHECKING:
    from aurora import Batch

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
    weights: Aurora_Weights | None = None,
    *args: Any,
    data_vars: Sequence[str] | None = None,
    variable_mapping: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Aurora model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.13063

    This model requires the following additional library to be installed:

    * `microsoft-aurora <https://pypi.org/project/microsoft-aurora/>`_ to load the models.

    .. versionadded:: 0.8

    .. versionadded:: 0.11
       The *data_vars* and *variable_mapping* parameters.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to ``aurora.Aurora``
        data_vars: Dataset variable names to predict. When supplied, enable weather
            dictionary inputs and prediction dictionaries. Otherwise, use Aurora's
            native batch interface.
        variable_mapping: Mapping from dataset names to Aurora names for surface,
            atmospheric, and static variables when *data_vars* is supplied.
            Unmapped names are kept unchanged.
            Weather dictionaries use surface inputs of shape ``(B, T, H, W)``,
            atmospheric inputs of shape ``(B, T, L, H, W)``, and shared static
            inputs of shape ``(H, W)``. Times are Unix nanoseconds, with the last
            input time used for each batch element. Predictions use dataset names
            and retain one time step.
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

    if data_vars is not None:
        variable_mapping = variable_mapping or {}
        model.register_forward_pre_hook(
            partial(_to_aurora_batch, variable_mapping=variable_mapping)
        )
        model.register_forward_hook(
            partial(
                _from_aurora_batch,
                data_vars=data_vars,
                variable_mapping=variable_mapping,
            )
        )
    return cast(nn.Module, model)


def _to_aurora_batch(
    module: nn.Module, inputs: tuple[Sample], variable_mapping: Mapping[str, str]
) -> 'Batch':
    """Convert a weather dictionary to Aurora's batch object."""
    batch = inputs[0]
    aurora = lazy_import('aurora')
    surface: Sample = {}
    atmosphere: Sample = {}
    static: Sample = {}
    for source, value in batch.items():
        if source in ('latitude', 'longitude', 'level', 'time'):
            continue
        name = variable_mapping.get(source, source)
        if value.ndim == 2:
            static[name] = value
        elif value.ndim == 4:
            surface[name] = value
        elif value.ndim == 5:
            atmosphere[name] = value
    metadata = aurora.Metadata(
        lat=batch['latitude'][0],
        lon=batch['longitude'][0],
        atmos_levels=tuple(batch['level'][0].tolist()),
        time=tuple(
            datetime.fromtimestamp(time / 1e9, tz=UTC)
            for time in batch['time'][:, -1].tolist()
        ),
    )
    return aurora.Batch(surface, static, atmosphere, metadata)


def _from_aurora_batch(
    module: nn.Module,
    inputs: tuple['Batch'],
    output: 'Batch',
    data_vars: Sequence[str],
    variable_mapping: Mapping[str, str],
) -> Sample:
    """Return Aurora predictions under their dataset variable names."""
    predictions = output.surf_vars | output.atmos_vars
    return {
        source: predictions[variable_mapping.get(source, source)]
        for source in data_vars
    }
