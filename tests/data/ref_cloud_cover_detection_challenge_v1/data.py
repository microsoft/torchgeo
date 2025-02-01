#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 2
DTYPE = np.uint16

np.random.seed(0)

splits = {'train': 'public', 'test': 'private'}
chip_ids = ['aaaa']
all_bands = ['B02', 'B03', 'B04', 'B08']
profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_epsg(32753),
    'transform': Affine(10.0, 0.0, 777760.0, 0.0, -10.0, 6735270.0),
}
Z = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE)

for split, directory in splits.items():
    for chip_id in chip_ids:
        path = os.path.join(directory, f'{split}_features', chip_id)
        os.makedirs(path, exist_ok=True)
        for band in all_bands:
            with rasterio.open(os.path.join(path, f'{band}.tif'), 'w', **profile) as f:
                f.write(Z, 1)
        path = os.path.join(directory, f'{split}_labels')
        os.makedirs(path, exist_ok=True)
        with rasterio.open(os.path.join(path, f'{chip_id}.tif'), 'w', **profile) as f:
            f.write(Z, 1)
