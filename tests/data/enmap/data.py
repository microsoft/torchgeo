#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
DTYPE = 'int16'

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'nodata': -32768.0,
    'width': SIZE,
    'height': SIZE,
    'count': 224,
    'crs': CRS.from_epsg(32640),
    'transform': Affine(30.0, 0.0, 283455.0, 0.0, -30.0, 2786715.0),
}

filename = 'ENMAP01-____L2A-DT0000001053_20220611T072305Z_002_V010400_20231221T134421Z-SPECTRAL_IMAGE_COG.tiff'

Z = np.random.randint(
    np.iinfo(DTYPE).min, np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE
)
with rasterio.open(filename, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)
