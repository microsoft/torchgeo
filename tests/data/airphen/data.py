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
    'dtype': 'uint16',
    'width': SIZE,
    'height': SIZE,
    'count': 6,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(
        4.497249999999613e-07,
        0.0,
        12.567765446921205,
        0.0,
        -4.4972499999996745e-07,
        47.42974580435403,
    ),
}

Z = np.random.randint(
    np.iinfo(profile['dtype']).max, size=(SIZE, SIZE), dtype=profile['dtype']
)

with rasterio.open('zoneA_B_R_NIR.tif', 'w', **profile) as src:
    for i in range(profile['count']):
        src.write(Z, i + 1)
