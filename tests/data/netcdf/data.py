#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import xarray as xr

SIZE = 32
PERIODS = 3

longitude = np.linspace(0, 90, SIZE)
latitude = np.linspace(0, 90, SIZE)
time = pd.date_range('2025-08-01', periods=PERIODS)

np.random.seed(0)
temperature = np.random.rand(PERIODS, SIZE, SIZE)
pressure = np.random.rand(PERIODS, SIZE, SIZE)

data_vars = {
    'temperature': (('time', 'latitude', 'longitude'), temperature),
    'pressure': (('time', 'latitude', 'longitude'), pressure),
}
coords = {'longitude': longitude, 'latitude': latitude, 'time': time}

ds = xr.Dataset(data_vars, coords)
ds.to_netcdf('era5.nc')
