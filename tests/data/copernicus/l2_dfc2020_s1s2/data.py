#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
filename = 'ROIs0000_test_dfc_0_p609.tif'

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'width': SIZE,
    'height': SIZE,
    'crs': CRS.from_wkt("""
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AXIS["Latitude",NORTH],
    AXIS["Longitude",EAST],
    AUTHORITY["EPSG","4326"]]
    """),
    'transform': Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
}

# Sentinel-1
directory = os.path.join('dfc2020_s1s2', 's1')
os.makedirs(directory, exist_ok=True)
profile['count'] = 2
profile['dtype'] = 'float64'
Z = np.random.rand(profile['height'], profile['width'])
path = os.path.join(directory, filename.replace('dfc', 's1'))
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Sentinel-2
directory = os.path.join('dfc2020_s1s2', 's2')
os.makedirs(directory, exist_ok=True)
profile['count'] = 13
profile['dtype'] = 'uint16'
Z = np.random.randint(
    np.iinfo(profile['dtype']).min,
    np.iinfo(profile['dtype']).max,
    size=(profile['height'], profile['width']),
    dtype=profile['dtype'],
)
path = os.path.join(directory, filename.replace('dfc', 's2'))
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Mask
directory = os.path.join('dfc2020_s1s2', 'dfc')
os.makedirs(directory, exist_ok=True)
profile['count'] = 1
profile['dtype'] = 'int32'
Z = np.random.randint(
    1, 11, size=(profile['height'], profile['width']), dtype=profile['dtype']
)
path = os.path.join(directory, filename)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Splits
for split in ['train', 'val', 'test']:
    with open(os.path.join('dfc2020_s1s2', f'dfc-{split}-new.csv'), 'w') as f:
        f.write(f'{filename}\n')

    # Zip
    shutil.make_archive('dfc2020', 'zip', '.', 'dfc2020_s1s2')
