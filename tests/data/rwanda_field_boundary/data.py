#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

dates = ('2021_03', '2021_04', '2021_08', '2021_10', '2021_11', '2021_12')
all_bands = ('B01', 'B02', 'B03', 'B04')

SIZE = 32
DTYPE = np.uint16
NUM_SAMPLES = 1
np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_epsg(3857),
    'transform': Affine(
        4.77731426716, 0.0, 3374518.037700199, 0.0, -4.77731426716, -168438.54642526805
    ),
}
Z = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE)

for sample in range(NUM_SAMPLES):
    for split in ['train', 'test']:
        for date in dates:
            path = os.path.join('source', split, date)
            os.makedirs(path, exist_ok=True)
            for band in all_bands:
                file = os.path.join(path, f'{sample:02}_{band}.tif')
                with rasterio.open(file, 'w', **profile) as src:
                    src.write(Z, 1)

    path = os.path.join('labels', 'train')
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f'{sample:02}.tif')
    with rasterio.open(file, 'w', **profile) as src:
        src.write(Z, 1)
