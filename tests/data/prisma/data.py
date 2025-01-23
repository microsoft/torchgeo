#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)


files = [
    'PRS_L0S_EO_NRT_20191215092453_20191215092457_0001.tif',
    'PRS_L1_STD_OFFL_20191215092453_20191215092457_0002.tif',
    'PRS_L2D_STD_20191215092453_20191215092457_0003.tif',
    'PRS_CF_AX_FDP_REPR_20191215092453_20191215092457_0004_0.tif',
]


for file in files:
    res = 10
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'count': 239,
        'crs': CRS.from_epsg(32634),
        'transform': Affine(
            29.974884647651006, 0.0, 718687.5, 0.0, -29.97457627118644, 4503407.5
        ),
        'height': SIZE,
        'width': SIZE,
    }

    Z = np.random.randint(
        np.iinfo(profile['dtype']).max, size=(SIZE, SIZE), dtype=profile['dtype']
    )

    with rasterio.open(file, 'w', **profile) as src:
        for i in range(profile['count']):
            src.write(Z, i + 1)
