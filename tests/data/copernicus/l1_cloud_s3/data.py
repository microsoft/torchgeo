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
filename = 'S3A_OL_1_EFR____20200703T154217_20200703T154517_20200704T201053_0179_060_111_2160_LN1_O_NT_002_01_01.tif'

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
    'transform': Affine(
        0.00411155313463515, 0.0, -90.345373, 0.0, -0.003155200244498778, 52.455946
    ),
}

# Image
directory = os.path.join('cloud_s3', 's3_olci')
os.makedirs(directory, exist_ok=True)
profile['count'] = 21
profile['dtype'] = 'float32'
Z = np.random.random(size=(profile['height'], profile['width']))
path = os.path.join(directory, filename)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Mask (binary)
directory = os.path.join('cloud_s3', 'cloud_binary')
os.makedirs(directory, exist_ok=True)
profile['count'] = 1
profile['dtype'] = 'uint8'
Z = np.random.randint(
    3, size=(profile['height'], profile['width']), dtype=profile['dtype']
)
path = os.path.join(directory, filename)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Mask (multi)
directory = os.path.join('cloud_s3', 'cloud_multi')
os.makedirs(directory, exist_ok=True)
profile['count'] = 1
profile['dtype'] = 'uint8'
Z = np.random.randint(
    6, size=(profile['height'], profile['width']), dtype=profile['dtype']
)
path = os.path.join(directory, filename)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Splits
filename = filename[: filename.index('.')]
for split in ['train', 'val', 'test']:
    with open(os.path.join('cloud_s3', f'{split}.csv'), 'w') as csv:
        csv.write(f'{filename}\n')

# Zip
shutil.make_archive('cloud_s3', 'zip', '.', 'cloud_s3')
