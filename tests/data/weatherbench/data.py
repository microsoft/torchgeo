#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Generate a tiny WeatherBench2-like ERA5 Zarr fixture for unit tests.

The fixture mirrors the variable layout of the public WeatherBench2 ERA5 store
(`gs://weatherbench2/datasets/era5/...`) but at a tiny resolution so it is fast
to read and small enough to commit. We only ship ``data.py`` (not the resulting
store) because most tests build their own short-lived fixture in ``tmp_path``.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

SIZE = 8
PERIODS = 4
LEVELS = (50, 250, 500, 1000)


def make_dataset(
    start_date: str = '2023-01-01',
    *,
    descending_lat: bool = True,
    descending_lon: bool = False,
) -> xr.Dataset:
    """Build a synthetic ERA5-like xarray Dataset."""
    rng = np.random.default_rng(0)
    # Latitude is descending in WeatherBench2 by default; flip via flags to
    # exercise the opposite axis ordering.
    lon_pair = (359.0, 0.0) if descending_lon else (0.0, 359.0)
    lat_pair = (90.0, -90.0) if descending_lat else (-90.0, 90.0)
    longitude = np.linspace(*lon_pair, SIZE).astype(np.float32)
    latitude = np.linspace(*lat_pair, SIZE).astype(np.float32)
    time = pd.date_range(start_date, periods=PERIODS, freq='6h')
    level = np.array(LEVELS, dtype=np.int32)

    surf_shape = (PERIODS, SIZE, SIZE)
    atmos_shape = (PERIODS, len(level), SIZE, SIZE)
    static_shape = (SIZE, SIZE)

    data_vars = {
        '2m_temperature': (
            ('time', 'latitude', 'longitude'),
            rng.standard_normal(surf_shape, dtype=np.float32) + 280,
        ),
        '10m_u_component_of_wind': (
            ('time', 'latitude', 'longitude'),
            rng.standard_normal(surf_shape, dtype=np.float32),
        ),
        '10m_v_component_of_wind': (
            ('time', 'latitude', 'longitude'),
            rng.standard_normal(surf_shape, dtype=np.float32),
        ),
        'mean_sea_level_pressure': (
            ('time', 'latitude', 'longitude'),
            rng.standard_normal(surf_shape, dtype=np.float32) * 100 + 101325,
        ),
        'temperature': (
            ('time', 'level', 'latitude', 'longitude'),
            rng.standard_normal(atmos_shape, dtype=np.float32) + 250,
        ),
        'u_component_of_wind': (
            ('time', 'level', 'latitude', 'longitude'),
            rng.standard_normal(atmos_shape, dtype=np.float32),
        ),
        'v_component_of_wind': (
            ('time', 'level', 'latitude', 'longitude'),
            rng.standard_normal(atmos_shape, dtype=np.float32),
        ),
        'specific_humidity': (
            ('time', 'level', 'latitude', 'longitude'),
            rng.uniform(0, 0.02, atmos_shape).astype(np.float32),
        ),
        'geopotential': (
            ('time', 'level', 'latitude', 'longitude'),
            rng.standard_normal(atmos_shape, dtype=np.float32) * 100 + 50000,
        ),
        'land_sea_mask': (
            ('latitude', 'longitude'),
            rng.uniform(0, 1, static_shape).astype(np.float32),
        ),
        'soil_type': (
            ('latitude', 'longitude'),
            rng.integers(0, 7, static_shape).astype(np.float32),
        ),
        'geopotential_at_surface': (
            ('latitude', 'longitude'),
            rng.standard_normal(static_shape, dtype=np.float32) * 1000,
        ),
    }
    coords = {
        'longitude': longitude,
        'latitude': latitude,
        'time': time,
        'level': level,
    }
    return xr.Dataset(data_vars, coords)


def main(
    out: str,
    start_date: str = '2023-01-01',
    *,
    descending_lat: bool = True,
    descending_lon: bool = False,
) -> None:
    """Write the fixture to disk."""
    make_dataset(
        start_date, descending_lat=descending_lat, descending_lon=descending_lon
    ).to_zarr(out, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='era5.zarr', help='output Zarr path')
    args = parser.parse_args()
    main(args.out)
