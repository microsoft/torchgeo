#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

filename = 'Togo_Presto_embeddings_v2025_06_190000000000-0000000000.tif'

profile = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 128,
    'crs': CRS.from_epsg(25231),
    'transform': Affine(10.0, 0.0, 156130.0, 0.0, -10.0, 1233190.0),
    'blockxsize': 256,
    'blockysize': 256,
    'tiled': True,
    'compress': 'lzw',
    'interleave': 'pixel',
}

Z = np.random.randint(0, np.iinfo(np.uint16).max, size=(profile['count'], SIZE, SIZE))
with rasterio.open(filename, 'w', **profile) as src:
    src.write(Z)
