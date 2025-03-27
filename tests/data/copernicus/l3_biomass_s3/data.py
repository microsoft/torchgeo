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

np.random.seed(0)

location = 'S30E140_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv4.0_02_11'
files = [
    'S3A_20200319T233546_20200319T233846.tif',
    'S3B_20200514T234457_20200514T234757.tif',
]

profile = {
    'driver': 'GTiff',
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
}

# Images
directory = os.path.join('biomass_s3', 's3_olci', location)
os.makedirs(directory, exist_ok=True)
profile['width'] = SIZE // 3
profile['height'] = SIZE // 3
profile['transform'] = Affine(
    0.0026949458523585646,
    0.0,
    142.50604172686855,
    0.0,
    -0.0026949458523585646,
    -30.24807224687253,
)
profile['count'] = 21
profile['dtype'] = 'float32'
Z = np.random.random(size=(profile['height'], profile['width']))
for file in files:
    path = os.path.join(directory, file)
    with rio.open(path, 'w', **profile) as src:
        for i in range(1, profile['count'] + 1):
            src.write(Z, i)

# Mask
directory = os.path.join('biomass_s3', 'biomass')
os.makedirs(directory, exist_ok=True)
profile['width'] = SIZE
profile['height'] = SIZE
profile['transform'] = Affine(
    0.00088888888888, 0.0, 142.5066666666416, 0.0, -0.00088888888888, -30.25066666666416
)
profile['count'] = 1
profile['dtype'] = 'uint16'
Z = np.random.randint(100, size=(profile['height'], profile['width']), dtype=np.uint16)
path = os.path.join(directory, f'{location}.tif')
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Splits
directory = 'biomass_s3'
for split in ['train', 'val', 'test']:
    with open(os.path.join(directory, f'static_fnames-{split}.csv'), 'w') as f:
        f.write(f'{location},{files[1]}\n')

# Zip
shutil.make_archive(directory, 'zip', '.', directory)
