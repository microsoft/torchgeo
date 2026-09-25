#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import xarray as xr

rng = np.random.default_rng(seed=0)

# Dimensions
X = 60
Y = 30
Z = 4
T = 3

time = pd.date_range('1959-01-01', '2023-01-10T18:00:00', T)
latitude = np.linspace(90.0, -90.0, Y)
longitude = np.linspace(0.0, 359.8, X)
level = np.linspace(50, 1000, Z)

# 2D variables
land_sea_mask = xr.DataArray(
    rng.random((Y, X)),
    coords={'latitude': latitude, 'longitude': longitude},
    dims=('latitude', 'longitude'),
    name='land_sea_mask',
    attrs={
        'long_name': 'Land-sea mask',
        'short_name': 'lsm',
        'standard_name': 'land_binary_mask',
        'units': '(0 - 1)',
    },
)
soil_type = xr.DataArray(
    rng.random((Y, X)),
    coords={'latitude': latitude, 'longitude': longitude},
    dims=('latitude', 'longitude'),
    name='soil_type',
    attrs={'long_name': 'Soil type', 'short_name': 'slt', 'units': '~'},
)

# 3D variables
_2m_temperature = xr.DataArray(
    rng.random((T, Y, X)),
    coords={'time': time, 'latitude': latitude, 'longitude': longitude},
    dims=('time', 'latitude', 'longitude'),
    name='2m_temperature',
    attrs={'long_name': '2 metre temperature', 'short_name': 't2m', 'units': 'K'},
)
_10m_wind_speed = xr.DataArray(
    rng.random((T, Y, X)),
    coords={'time': time, 'latitude': latitude, 'longitude': longitude},
    dims=('time', 'latitude', 'longitude'),
    name='10m_wind_speed',
    attrs={},
)

# 4D variables
temperature = xr.DataArray(
    rng.random((T, Z, Y, X)),
    coords={'time': time, 'level': level, 'latitude': latitude, 'longitude': longitude},
    dims=('time', 'level', 'latitude', 'longitude'),
    name='temperature',
    attrs={
        'long_name': 'Temperature',
        'short_name': 't',
        'standard_name': 'air_temperature',
        'units': 'K',
    },
)
wind_speed = xr.DataArray(
    rng.random((T, Z, Y, X)),
    coords={'time': time, 'level': level, 'latitude': latitude, 'longitude': longitude},
    dims=('time', 'level', 'latitude', 'longitude'),
    name='wind_speed',
    attrs={},
)

# Combined dataset
dataset = xr.Dataset(
    {
        '10m_wind_speed': _10m_wind_speed,
        '2m_temperature': _2m_temperature,
        'land_sea_mask': land_sea_mask,
        'soil_type': soil_type,
        'temperature': temperature,
        'wind_speed': wind_speed,
    },
    coords={'time': time, 'latitude': latitude, 'longitude': longitude, 'level': level},
)
dataset.to_zarr('1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
