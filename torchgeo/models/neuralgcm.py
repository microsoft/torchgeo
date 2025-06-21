# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained NeuralGCM models."""

import pickle
from typing import Any, cast

import gcsfs
import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

# NeuralGCM operates on raw unnormalized data in ERA5 format
_neuralgcm_transforms = nn.Identity()

_neuralgcm_meta = {
    'dataset': 'ERA5',
    'model': None,
    'resolution': None,
    'architecture': 'Hybrid ML + physics atmospheric model',
    'publication': 'https://www.nature.com/articles/s41586-024-07744-y',
    'repo': 'https://github.com/neuralgcm/neuralgcm',
    'license': 'CC BY-SA 4.0',
    'units': 'various (pressure levels, temperature, etc.)',
}


class NeuralGCM_Weights(WeightsEnum):  # type: ignore[misc]
    """NeuralGCM weights.

    If you use this model in your research, please cite the following paper:

    * https://www.nature.com/articles/s41586-024-07744-y

    .. versionadded:: 0.8
    """

    DETERMINISTIC_0_7_DEG = Weights(
        url='gs://neuralgcm/models/v1/deterministic_0_7_deg.pkl',
        transforms=_neuralgcm_transforms,
        meta=_neuralgcm_meta | {'model': 'NeuralGCM07Deterministic', 'resolution': 0.7},
    )

    DETERMINISTIC_1_4_DEG = Weights(
        url='gs://neuralgcm/models/v1/deterministic_1_4_deg.pkl',
        transforms=_neuralgcm_transforms,
        meta=_neuralgcm_meta | {'model': 'NeuralGCM14Deterministic', 'resolution': 1.4},
    )

    DETERMINISTIC_2_8_DEG = Weights(
        url='gs://neuralgcm/models/v1/deterministic_2_8_deg.pkl',
        transforms=_neuralgcm_transforms,
        meta=_neuralgcm_meta | {'model': 'NeuralGCM28Deterministic', 'resolution': 2.8},
    )

    STOCHASTIC_1_4_DEG = Weights(
        url='gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl',
        transforms=_neuralgcm_transforms,
        meta=_neuralgcm_meta | {'model': 'NeuralGCM14Stochastic', 'resolution': 1.4},
    )

    STOCHASTIC_PRECIP_2_8_DEG = Weights(
        url='gs://neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl',
        transforms=_neuralgcm_transforms,
        meta=_neuralgcm_meta
        | {'model': 'NeuralGCM28StochasticPrecip', 'resolution': 2.8},
    )

    STOCHASTIC_EVAP_2_8_DEG = Weights(
        url='gs://neuralgcm/models/v1_precip/stochastic_evap_2_8_deg.pkl',
        transforms=_neuralgcm_transforms,
        meta=_neuralgcm_meta
        | {'model': 'NeuralGCM28StochasticEvap', 'resolution': 2.8},
    )


def neuralgcm(
    weights: NeuralGCM_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """NeuralGCM PressureLevelModel.

    If you use this model in your research, please cite the following paper:

    * https://www.nature.com/articles/s41586-024-07744-y

    This model requires the following additional library to be installed:

    * `neuralgcm <https://pypi.org/project/neuralgcm/>`_ to load the models.

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to ``neuralgcm.PressureLevelModel``
        **kwargs: Additional keyword arguments to pass to ``neuralgcm.PressureLevelModel``

    Returns:
        A NeuralGCM PressureLevelModel.
    """
    lazy_import('neuralgcm')
    import neuralgcm

    if weights is None:
        msg = 'NeuralGCM requires pre-trained weights. Please specify a weights enum.'
        raise ValueError(msg)

    # Load checkpoint from Google Cloud Storage
    gcs = gcsfs.GCSFileSystem(token='anon')
    with gcs.open(weights.url, 'rb') as f:
        checkpoint = pickle.load(f)

    model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)

    return cast(nn.Module, model)
