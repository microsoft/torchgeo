#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_epsg(3857),
    'transform': Affine(3.0, 0.0, -1333.7802539161332, 0.0, -3.0, 1120234.423089223),
}

Z = np.random.choice([0, 255], size=(SIZE, SIZE))

with rasterio.open('GBM_v1_e000_n10_e005_n05.tif', 'w', **profile) as src:
    src.write(Z, 1)
