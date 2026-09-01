#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

file = 'embed_map_310k.tif'

profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 768,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(0.25, 0.0, -180.125, 0.0, -0.25, 90.125),
    'blockxsize': 256,
    'blockysize': 256,
    'tiled': True,
    'compress': 'deflate',
    'interleave': 'pixel',
}

Z = np.random.random(size=(profile['count'], SIZE, SIZE)) * 2 - 1
with rasterio.open(file, 'w', **profile) as src:
    src.write(Z)
