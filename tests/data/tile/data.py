#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

np.random.seed(0)

os.makedirs('images', exist_ok=True)
os.makedirs('masks', exist_ok=True)

sizes = [(64, 64), (32, 48), (100, 80)]
files = ['tile_0', 'tile_1', 'tile_2']

for file, (height, width) in zip(files, sizes):
    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'width': width,
        'height': height,
        'count': 3,
        'crs': CRS.from_epsg(4326),
        'transform': from_bounds(0, 0, 1, 1, width, height),
    }

    with rio.open(os.path.join('images', f'{file}.tif'), 'w', **profile) as f:
        for i in range(1, 4):
            arr = np.random.randint(256, size=(height, width), dtype=np.uint8)
            f.write(arr, i)

    profile['count'] = 1
    with rio.open(os.path.join('masks', f'{file}.tif'), 'w', **profile) as f:
        arr = np.random.randint(5, size=(height, width), dtype=np.uint8)
        f.write(arr, 1)
