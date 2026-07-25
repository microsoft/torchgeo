# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# https://github.com/gastruc/UniverSat

"""Pre-trained UniverSat models."""

from typing import Any, cast

import torch
import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

# UniverSat is loaded through its Torch Hub entrypoint (``hubconf.py``) rather than
# being vendored here, so the entire model implementation lives in the upstream
# repository. The repository is pinned to a specific commit for reproducibility.
_universat_repo = 'gastruc/UniverSat'
_universat_ref = 'f6df2eec54955b0f7524cc95fe21a5e80c0239d9'

# UniverSat operates on raw reflectance/backscatter values; no normalization is
# baked into the released weights.
_universat_transforms = nn.Identity()

_universat_meta = {
    'dataset': 'GeoPlexV2',
    'model': 'universat',
    'architecture': 'Universal Patch Encoder + ViT',
    'publication': 'https://arxiv.org/abs/2606.23503',
    'repo': 'https://github.com/gastruc/UniverSat',
    'hf_repo': 'g-astruc/UniverSat',
    'license': 'MIT',
    'model_size': None,
}


class UniverSat_Base_Weights(WeightsEnum):
    """UniverSat-Base weights.

    UniverSat is a resolution- and modality-agnostic transformer for Earth
    observation, jointly trained on 13 sensors from 7 datasets (GeoPlexV2). The
    released checkpoint is a Base model (~201M parameters).

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2606.23503

    .. versionadded:: 0.10
    """

    GEOPLEX = Weights(
        url='https://huggingface.co/g-astruc/UniverSat/resolve/b6456ee87a162128ea5600cf45d2767a6e48f2f9/model.safetensors',
        transforms=_universat_transforms,
        meta=_universat_meta | {'model_size': 'base'},
    )


def universat(
    weights: UniverSat_Base_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """UniverSat-Base model.

    UniverSat (AnySat v2) is a multimodal, multi-resolution Earth observation
    encoder. A single set of weights processes arbitrary combinations of
    sensors, spatial resolutions, spectral bands, and time-series depths
    without resampling or channel selection. The output resolution is chosen at
    inference time and decoupled from the input patch size, down to per-pixel
    features.

    The model is loaded through its Torch Hub entrypoint
    (https://github.com/gastruc/UniverSat/blob/main/hubconf.py); the
    implementation is not vendored into TorchGeo. The returned module exposes a
    high-level ``encode`` method that auto-fills per-modality metadata
    (wavelengths, physical resolution, sub-patch factors) from a built-in
    registry, so a forward pass only needs a dict of tensors.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2606.23503

    This model requires the following additional libraries to be installed:

    * `huggingface_hub <https://pypi.org/project/huggingface-hub/>`_ and
      `safetensors <https://pypi.org/project/safetensors/>`_: to load the
      pre-trained weights.

    .. versionadded:: 0.10

    Args:
        weights: Pre-trained weights to use. If ``None``, the model is randomly
            initialized.
        *args: Additional arguments to pass to the ``universat`` Torch Hub
            entrypoint.
        **kwargs: Additional keyword arguments to pass to the ``universat``
            Torch Hub entrypoint (e.g. ``size``, ``modalities_dict``).

    Returns:
        A UniverSat-Base model.
    """
    model = torch.hub.load(
        f'{_universat_repo}:{_universat_ref}',
        'universat',
        *args,
        pretrained=weights is not None,
        trust_repo=True,
        **kwargs,
    )
    return cast(nn.Module, model)
