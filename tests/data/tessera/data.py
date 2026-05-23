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

directory = os.path.join('global_0.1_degree_representation', '2024', 'grid_0.05_51.35')
filename = 'grid_0.05_51.35_2024.tiff'

profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 128,
    'crs': CRS.from_epsg(32631),
    'transform': Affine(10.0, 0.0, 290872.40803907975, 0.0, -10.0, 5698579.144861946),
    'blockxsize': 256,
    'blockysize': 256,
    'tiled': True,
    'compress': 'lzw',
    'interleave': 'pixel',
}

Z = np.random.random(size=(profile['count'], SIZE, SIZE)) * 2 - 1
os.makedirs(directory, exist_ok=True)
with rasterio.open(os.path.join(directory, filename), 'w', **profile) as src:
    src.write(Z)
