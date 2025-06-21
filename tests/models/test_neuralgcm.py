# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from torchgeo.models import NeuralGCM_Weights, neuralgcm


class TestNeuralGCM:
    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match='Please specify a weights enum'):
            neuralgcm()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'weights',
        [
            NeuralGCM_Weights.DETERMINISTIC_0_7_DEG,
            NeuralGCM_Weights.DETERMINISTIC_1_4_DEG,
            NeuralGCM_Weights.DETERMINISTIC_2_8_DEG,
            NeuralGCM_Weights.STOCHASTIC_1_4_DEG,
            NeuralGCM_Weights.STOCHASTIC_PRECIP_2_8_DEG,
            NeuralGCM_Weights.STOCHASTIC_EVAP_2_8_DEG,
        ],
    )
    def test_weights(self, weights: NeuralGCM_Weights) -> None:
        pytest.importorskip('neuralgcm', minversion='0.2.0')
        model = neuralgcm(weights=weights)
        assert hasattr(model, 'timestep')
        assert hasattr(model, 'data_coords')
        assert hasattr(model, 'model_coords')
        assert hasattr(model, 'input_variables')
        assert hasattr(model, 'forcing_variables')

    @pytest.mark.slow
    def test_default_weights(self) -> None:
        pytest.importorskip('neuralgcm', minversion='0.2.0')
        model = neuralgcm(weights=NeuralGCM_Weights.DETERMINISTIC_1_4_DEG)
        assert hasattr(model, 'timestep')
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        assert hasattr(model, 'advance')
        assert hasattr(model, 'unroll')

    @pytest.mark.slow
    def test_forecasting_integration(self) -> None:
        """Test NeuralGCM forecasting workflow using ERA5 data from GCS.

        Performs parts of the NeuralGCM quick start guide:
        https://neuralgcm.readthedocs.io/en/latest/inference_demo.html
        """
        pytest.importorskip('neuralgcm', minversion='0.2.0')
        pytest.importorskip('jax')
        pytest.importorskip('xarray')
        pytest.importorskip('gcsfs')

        import jax
        import numpy as np
        import xarray as xr
        from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

        # Load model
        model = neuralgcm(weights=NeuralGCM_Weights.DETERMINISTIC_2_8_DEG)

        # Load ERA5 data
        era5_path = (
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
        )
        full_era5 = xr.open_zarr(
            era5_path, chunks=None, storage_options=dict(token='anon')
        )

        # Select slice
        start_time = '2020-02-14'
        end_time = '2020-02-15'  # 1 day
        data_inner_steps = 24  # process every 24th hour

        sliced_era5 = (
            full_era5[model.input_variables + model.forcing_variables]
            .pipe(
                xarray_utils.selective_temporal_shift,
                variables=model.forcing_variables,
                time_shift='24 hours',
            )
            .sel(time=slice(start_time, end_time, data_inner_steps))
            .compute()
        )

        # Regrid to NeuralGCM's native resolution
        era5_grid = spherical_harmonic.Grid(
            latitude_nodes=full_era5.sizes['latitude'],
            longitude_nodes=full_era5.sizes['longitude'],
            latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
            longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
        )
        regridder = horizontal_interpolation.ConservativeRegridder(
            era5_grid, model.data_coords.horizontal, skipna=True
        )
        eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
        eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

        # Model workflow
        inner_steps = 24  # save model outputs once every 24 hours
        outer_steps = 2  # 2 steps = 24 hours forecast
        timedelta = np.timedelta64(1, 'h') * inner_steps
        times = np.arange(outer_steps) * inner_steps  # time axis in hours

        inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
        input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
        rng_key = jax.random.key(42)  # optional for deterministic models
        initial_state = model.encode(inputs, input_forcings, rng_key)
        all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

        final_state, predictions = model.unroll(
            initial_state,
            all_forcings,
            steps=outer_steps,
            timedelta=timedelta,
            start_with_input=True,
        )

        predictions_ds = model.data_to_xarray(predictions, times=times)

        # Verify the forecast worked
        assert predictions_ds is not None
        assert 'time' in predictions_ds.dims
        assert len(predictions_ds.time) == outer_steps
        assert 'latitude' in predictions_ds.dims
        assert 'longitude' in predictions_ds.dims

        # Check that key variables are present and have valid data
        expected_vars = [
            'geopotential',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
        ]
        for var in expected_vars:
            if var in predictions_ds:
                assert predictions_ds[var].shape[0] == outer_steps  # time dimension
                assert not np.isnan(predictions_ds[var].values).all()  # not all NaN

        # Test comparison with ERA5
        if len(eval_era5.time) >= outer_steps:
            target_trajectory = model.inputs_from_xarray(
                eval_era5.thin(time=(inner_steps // data_inner_steps)).isel(
                    time=slice(outer_steps)
                )
            )
            target_data_ds = model.data_to_xarray(target_trajectory, times=times)

            # Verify target data structure
            assert target_data_ds is not None
            assert len(target_data_ds.time) == len(predictions_ds.time)
