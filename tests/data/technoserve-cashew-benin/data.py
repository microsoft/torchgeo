#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

DTYPE = np.uint16
SIZE = 2

np.random.seed(0)

dates = ('00_20191105',)
all_bands = (
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B8A',
    'B09',
    'B11',
    'B12',
    'CLD',
)
profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_epsg(32631),
    'transform': Affine(
        10.002549584378608,
        0.0,
        440853.29890114715,
        0.0,
        -9.99842989423825,
        1012804.082877621,
    ),
}

for date in dates:
    os.makedirs(os.path.join('imagery', '00', date), exist_ok=True)
    for band in all_bands:
        Z = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE)
        path = os.path.join('imagery', '00', date, f'{date}_{band}_10m.tif')
        with rasterio.open(path, 'w', **profile) as src:
            src.write(Z, 1)
