#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

# Official GCP, Source Cooperative format
directory = os.path.join('2024', '10N')
filename = 'x086q72fv2f9q1x4a-0000000000-0000000000.tiff'

profile = {
    'driver': 'GTiff',
    'dtype': 'int8',
    'nodata': -128.0,
    'width': SIZE,
    'height': SIZE,
    'count': 64,
    'crs': CRS.from_epsg(32610),
    'transform': Affine(10.0, 0.0, 500000.0, 0.0, 10.0, 6062080.0),  # upside down
    'blockxsize': 1024,
    'blockysize': 1024,
    'tiled': True,
    'compress': 'zstd',
    'interleave': 'band',
}

dtype = profile['dtype']
size = (profile['count'], SIZE, SIZE)
Z = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, size=size, dtype=dtype)
os.makedirs(directory, exist_ok=True)
with rasterio.open(os.path.join(directory, filename), 'w', **profile) as src:
    src.write(Z)
with rasterio.open(filename, 'w', **profile) as src:
    src.write(Z)

# Major TOM Hugging Face format
directory = os.path.join('2024', 'U', '1', 'L', '7')
filename = '471U_587L.tif'

profile = {
    'driver': 'GTiff',
    'dtype': 'float64',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 64,
    'crs': CRS.from_epsg(32619),
    'transform': Affine(10.0, 0.0, 310455.1604040997, 0.0, -10.0, 4696576.002073958),
    'blockxsize': 256,
    'blockysize': 256,
    'tiled': True,
    'compress': 'deflate',
    'interleave': 'pixel',
}

Z = np.random.random(size=(profile['count'], SIZE, SIZE)) * 2 - 1
os.makedirs(directory, exist_ok=True)
with rasterio.open(os.path.join(directory, filename), 'w', **profile) as src:
    src.write(Z)
