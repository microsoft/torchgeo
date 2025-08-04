# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import cftime
import numpy as np
import pandas as pd
import xarray as xr

SIZE = 32

LATS: list[tuple[float]] = [(40, 42), (60, 62), (80, 82)]

LONS: list[tuple[float]] = [(-55, -50), (-5, 5), (80, 85)]

VAR_NAMES = ['zos', 'tos']

DIR = 'data'

CF_TIME = [True, False, True]

NUM_TIME_STEPS = 3


def create_rioxr_dataset(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cf_time: bool,
    var_name: str,
    filename: str,
):
    # Generate x and y coordinates
    lats = np.linspace(lat_min, lat_max, SIZE)
    lons = np.linspace(lon_min, lon_max, SIZE)

    if cf_time:
        times = [cftime.datetime(2000, 1, i + 1) for i in range(NUM_TIME_STEPS)]
    else:
        times = pd.date_range(start='2000-01-01', periods=NUM_TIME_STEPS, freq='D')

    # data with shape (time, x, y)
    data = np.random.rand(len(times), len(lons), len(lats))

    # Create the xarray dataset
    ds = xr.Dataset(
        data_vars={var_name: (('time', 'x', 'y'), data)},
        coords={'x': lons, 'y': lats, 'time': times},
    )
    ds['x'].attrs['units'] = 'degrees_east'
    ds['x'].attrs['crs'] = 'EPSG:4326'
    ds['y'].attrs['units'] = 'degrees_north'
    ds['y'].attrs['crs'] = 'EPSG:4326'
    ds.to_netcdf(path=filename)


if __name__ == '__main__':
    if os.path.isdir(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)
    for var_name in VAR_NAMES:
        for lats, lons, cf_time in zip(LATS, LONS, CF_TIME):
            path = os.path.join(DIR, f'{var_name}_{lats}_{lons}.nc')
            create_rioxr_dataset(
                lats[0], lats[1], lons[0], lons[1], cf_time, var_name, path
            )
