# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import dataclasses
import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from pytest import MonkeyPatch

from torchgeo.models import ACE2_Weights, ace2_climate_emulator

pytest.importorskip('fme')


class PlusOne(torch.nn.Module):
    """Trivial module that adds one to its input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def _save_fake_stepper(path: Path) -> None:
    """Create a minimal ACE2 stepper checkpoint for testing.

    Adapted from ``fme/ace/inference/test_inference.py::save_stepper``.
    """
    from fme.ace.registry import ModuleSelector
    from fme.ace.stepper import StepperConfig
    from fme.core.coordinates import (
        HybridSigmaPressureCoordinate,
        LatLonCoordinates,
    )
    from fme.core.dataset.data_typing import VariableMetadata
    from fme.core.dataset_info import DatasetInfo
    from fme.core.normalizer import (
        NetworkAndLossNormalizationConfig,
        NormalizationConfig,
    )
    from fme.core.ocean import OceanConfig
    from fme.core.step.single_module import SingleModuleStepConfig
    from fme.core.step.step import StepSelector

    in_names = ['prog', 'sst', 'forcing_var', 'DSWRFtoa']
    out_names = ['prog', 'sst', 'ULWRFtoa', 'USWRFtoa']
    all_names = list(set(in_names) | set(out_names))

    config = StepperConfig(
        step=StepSelector(
            type='single_module',
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type='prebuilt', config={'module': PlusOne()}
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={name: 0.0 for name in all_names},
                            stds={name: 1.0 for name in all_names},
                        ),
                    ),
                    ocean=OceanConfig(
                        surface_temperature_name='sst',
                        ocean_fraction_name='ocean_fraction',
                    ),
                ),
            ),
        ),
    )
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(16, dtype=torch.float32),
        lon=torch.zeros(32, dtype=torch.float32),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(4), bk=torch.arange(4)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=datetime.timedelta(hours=6),
        variable_metadata={
            'prog': VariableMetadata(
                units='m',
                long_name='a prognostic variable',
            ),
        },
    )
    stepper = config.get_stepper(dataset_info=dataset_info)
    torch.save({'stepper': stepper.get_state()}, path)


class TestACE2:
    @pytest.fixture(params=[*ACE2_Weights])
    def weights(self, request: pytest.FixtureRequest) -> ACE2_Weights:
        return request.param  # type: ignore[no-any-return]

    @pytest.fixture
    def fake_checkpoint(self, tmp_path: Path) -> str:
        ckpt_path = tmp_path / 'fake_stepper'
        _save_fake_stepper(ckpt_path)
        return str(ckpt_path)

    def test_ace2_climate_emulator_checkpoint_path(
        self, fake_checkpoint: str
    ) -> None:
        model = ace2_climate_emulator(checkpoint_path=fake_checkpoint)
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'stepper')

    def test_ace2_climate_emulator_weights(
        self,
        fake_checkpoint: str,
        weights: ACE2_Weights,
        monkeypatch: MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            'torchgeo.models.ace2._download_ace2_checkpoint',
            lambda repo_id, filename: fake_checkpoint,
        )
        model = ace2_climate_emulator(weights=weights)
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'stepper')

    def test_forward_raises(self, fake_checkpoint: str) -> None:
        model = ace2_climate_emulator(checkpoint_path=fake_checkpoint)
        with pytest.raises(NotImplementedError, match='ACE2 does not support'):
            model(torch.randn(1, 4, 16, 32))

    def test_no_weights_or_path_raises(self) -> None:
        with pytest.raises(ValueError, match='Either weights or checkpoint_path'):
            ace2_climate_emulator()

    @pytest.mark.slow
    def test_ace2_download(self, weights: ACE2_Weights) -> None:
        ace2_climate_emulator(weights=weights)

    @pytest.mark.slow
    def test_ace2_autoregressive_forecast(self, weights: ACE2_Weights) -> None:
        """Download a real checkpoint and run one autoregressive forward step."""
        from fme.ace.data_loading.batch_data import BatchData, PrognosticState

        model = ace2_climate_emulator(weights=weights)
        stepper = model.stepper
        stepper.set_eval()

        # Extract metadata from the loaded stepper
        prognostic_names = stepper.prognostic_names
        all_names = stepper.config.all_names
        n_ic = stepper.n_ic_timesteps
        grid_shape = stepper.training_dataset_info.horizontal_coordinates.shape

        n_forward_steps = 1
        n_batch = 1
        n_forcing_timesteps = n_ic + n_forward_steps

        # Build initial condition (prognostic variables only)
        ic_data = {
            name: torch.randn(n_batch, n_ic, *grid_shape)
            for name in prognostic_names
        }
        ic_time = xr.DataArray(
            np.zeros((n_batch, n_ic)), dims=['sample', 'time']
        )
        initial_condition = PrognosticState(
            BatchData.new_on_cpu(data=ic_data, time=ic_time)
        )

        # Build forcing data (all variable names; predict() subsets internally)
        forcing_data = {
            name: torch.randn(n_batch, n_forcing_timesteps, *grid_shape)
            for name in all_names
        }
        forcing_time = xr.DataArray(
            np.arange(n_batch * n_forcing_timesteps, dtype=float).reshape(
                n_batch, n_forcing_timesteps
            ),
            dims=['sample', 'time'],
        )
        forcing = BatchData.new_on_cpu(data=forcing_data, time=forcing_time)

        # Run one autoregressive forward step
        with torch.no_grad():
            output, new_state = stepper.predict(
                initial_condition,
                forcing,
                compute_derived_variables=False,
                compute_derived_forcings=False,
            )

        # Verify output structure and shapes
        assert isinstance(output, BatchData)
        assert isinstance(new_state, PrognosticState)
        for name in stepper.out_names:
            assert name in output.data
            t = output.data[name]
            assert t.shape[0] == n_batch
            assert t.shape[1] == n_forward_steps
            assert t.shape[2:] == grid_shape

