#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
DTYPE = np.uint8

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'width': SIZE,
    'height': SIZE,
    'count': 3,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(
        2.1457672119140625e-05,
        0.0,
        -87.626953125,
        0.0,
        -2.0629065249348766e-05,
        15.977172621632805,
    ),
}

os.makedirs('source', exist_ok=True)
os.makedirs('labels', exist_ok=True)

files = [
    '20160928_153233_0e16_16816-29821-16',
    '20160928_153233_0e16_16816-29824-16',
    '20160928_153233_0e16_16816-29825-16',
    '20160928_153233_0e16_16816-29828-16',
    '20160928_153233_0e16_16816-29829-16',
]
for file in files:
    with rio.open(os.path.join('source', f'{file}.tif'), 'w', **profile) as f:
        for i in range(1, 4):
            Z = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE)
            f.write(Z, i)

    count = np.random.randint(5)
    x = np.random.randint(SIZE, size=count)
    y = np.random.randint(SIZE, size=count)
    dx = np.random.randint(5, size=count)
    dy = np.random.randint(5, size=count)
    label = np.ones(count)
    Z = np.stack([x, y, x + dx, y + dy, label], axis=-1)
    np.save(os.path.join('labels', f'{file}.npy'), Z)
