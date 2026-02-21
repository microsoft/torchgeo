# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ACE2 models."""

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

# ACE2 operates on the raw unnormalized data.
_ace2_transforms = nn.Identity()

_ace2_meta: dict[str, object] = {
    'dataset': None,
    'architecture': 'SFNO',
    'hf_repo': None,
    'filename': None,
    'resolution': 1.0,
    'timestep': '6 hours',
    'prognostic_vars': (
        'PRESsfc',
        'surface_temperature',
        'TMP2m',
        'Q2m',
        'UGRD10m',
        'VGRD10m',
        'air_temperature_0',
        'air_temperature_1',
        'air_temperature_2',
        'air_temperature_3',
        'air_temperature_4',
        'air_temperature_5',
        'air_temperature_6',
        'air_temperature_7',
        'specific_total_water_0',
        'specific_total_water_1',
        'specific_total_water_2',
        'specific_total_water_3',
        'specific_total_water_4',
        'specific_total_water_5',
        'specific_total_water_6',
        'specific_total_water_7',
        'eastward_wind_0',
        'eastward_wind_1',
        'eastward_wind_2',
        'eastward_wind_3',
        'eastward_wind_4',
        'eastward_wind_5',
        'eastward_wind_6',
        'eastward_wind_7',
        'northward_wind_0',
        'northward_wind_1',
        'northward_wind_2',
        'northward_wind_3',
        'northward_wind_4',
        'northward_wind_5',
        'northward_wind_6',
        'northward_wind_7',
    ),
    'forcing_vars': (
        'DSWRFtoa',
        'HGTsfc',
        'land_fraction',
        'ocean_fraction',
        'sea_ice_fraction',
        'surface_temperature',
        'global_mean_co2',
    ),
    'publication': 'https://arxiv.org/abs/2310.02074',
    'repo': 'https://github.com/ai2cm/ace',
    'license': 'Apache-2.0',
}


class ACE2_Weights(WeightsEnum):  # type: ignore[misc]
    """ACE2 pre-trained model weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2310.02074

    .. versionadded:: 0.8
    """

    ACE2_ERA5 = Weights(
        url='https://huggingface.co/allenai/ACE2-ERA5/resolve/main/ace2_era5_ckpt.tar',
        transforms=_ace2_transforms,
        meta=_ace2_meta
        | {
            'hf_repo': 'allenai/ACE2-ERA5',
            'filename': 'ace2_era5_ckpt.tar',
            'dataset': 'ERA5',
        },
    )

    ACE2_EAMv3 = Weights(
        url='https://huggingface.co/allenai/ACE2-EAMv3/resolve/main/ace2_EAMv3_ckpt.tar',
        transforms=_ace2_transforms,
        meta=_ace2_meta
        | {
            'hf_repo': 'allenai/ACE2-EAMv3',
            'filename': 'ace2_EAMv3_ckpt.tar',
            'dataset': 'EAMv3',
        },
    )


def _download_ace2_checkpoint(repo_id: str, filename: str) -> str:
    """Download an ACE2 checkpoint from HuggingFace.

    This function requires the following additional library to be installed:

    * `huggingface_hub <https://pypi.org/project/huggingface-hub/>`_

    Args:
        repo_id: HuggingFace repository ID (e.g. ``allenai/ACE2-ERA5``).
        filename: Name of the checkpoint file (e.g. ``ace2_era5_ckpt.tar``).

    Returns:
        Path to the downloaded checkpoint file.
    """
    huggingface_hub = lazy_import('huggingface_hub')
    return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)


class _ACE2Wrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around an ``fme.ace.stepper.Stepper``.

    ACE2 uses a ``Stepper`` abstraction rather than a standard ``nn.Module``.
    This wrapper exposes the stepper's internal neural network modules so that
    TorchGeo's weight-management API can treat it like any other model.

    The :meth:`forward` method is intentionally not implemented because ACE2
    inference requires forcing data, time stepping, and other context that
    cannot be captured by a single ``forward(x)`` call. Use
    :func:`run_ace2_inference` for full inference.
    """

    def __init__(self, stepper: object) -> None:
        super().__init__()
        self._stepper = stepper
        # Register the stepper's nn.Modules so that .parameters() works.
        for i, mod in enumerate(stepper.modules):  # type: ignore[union-attr]
            self.add_module(f'stepper_module_{i}', mod)

    @property
    def stepper(self) -> object:
        """Return the underlying ``fme.ace.stepper.Stepper`` instance."""
        return self._stepper

    def forward(self, *args: object, **kwargs: object) -> None:
        """Not implemented.

        Raises:
            NotImplementedError: Always. Use :func:`run_ace2_inference` instead.
        """
        raise NotImplementedError(
            'ACE2 does not support a simple forward() call. '
            'Use torchgeo.models.run_ace2_inference() for full inference.'
        )


def ace2_climate_emulator(
    weights: ACE2_Weights | None = None,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """ACE2 climate emulator model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2310.02074

    This model requires the following additional library to be installed:

    * `fme <https://pypi.org/project/fme/>`_ to load the models.

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use. If provided,
            *checkpoint_path* is ignored.
        checkpoint_path: Local path to a stepper checkpoint file. Only used
            when *weights* is ``None``.

    Returns:
        An ACE2 model wrapped as an ``nn.Module``.

    Raises:
        ValueError: If neither *weights* nor *checkpoint_path* is provided.
    """
    fme_stepper = lazy_import('fme.ace.stepper')

    if weights is not None:
        repo_id: str = weights.meta['hf_repo']
        filename: str = weights.meta['filename']
        checkpoint_path = _download_ace2_checkpoint(repo_id, filename)

    if checkpoint_path is None:
        raise ValueError(
            'Either weights or checkpoint_path must be provided '
            'to load an ACE2 model.'
        )

    stepper = fme_stepper.load_stepper(checkpoint_path)
    return _ACE2Wrapper(stepper)


def run_ace2_inference(yaml_config: str) -> None:
    """Run ACE2 inference from a YAML configuration file.

    This delegates to ``fme.ace.inference.main()``, which handles initial
    conditions, forcing data, time-stepping, and output writing.

    This function requires the following additional library to be installed:

    * `fme <https://pypi.org/project/fme/>`_ to run inference.

    .. versionadded:: 0.8

    Args:
        yaml_config: Path to a YAML inference configuration file compatible
            with ``fme.ace.inference.InferenceConfig``.
    """
    fme_inference = lazy_import('fme.ace.inference.inference')
    fme_inference.main(yaml_config)

