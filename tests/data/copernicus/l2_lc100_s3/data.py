#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32

np.random.seed(0)

location = '0200599_-70.25_-55.25'
files = [
    'S3A_20190507T135736_20190507T135744.tif',
    'S3B_20190810T135525_20190810T135539.tif',
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
directory = os.path.join('lc100_s3', 's3_olci', location)
os.makedirs(directory, exist_ok=True)
profile['width'] = SIZE // 3
profile['height'] = SIZE // 3
profile['transform'] = Affine(
    0.0026949458523585646,
    0.0,
    -69.62662104153587,
    0.0,
    -0.0026949458523585646,
    -55.37305242841143,
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
directory = os.path.join('lc100_s3', 'lc100')
os.makedirs(directory, exist_ok=True)
profile['width'] = SIZE
profile['height'] = SIZE
profile['transform'] = Affine(
    0.0008983152841195215,
    0.0,
    -70.37581598849155,
    0.0,
    -0.0008983152841195215,
    -55.12421909471032,
)
profile['count'] = 1
profile['dtype'] = 'uint8'
values = [
    0,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    111,
    112,
    113,
    114,
    115,
    116,
    121,
    122,
    123,
    124,
    125,
    126,
    200,
]
Z = np.random.choice(values, size=(profile['height'], profile['width']))
path = os.path.join(directory, f'{location}.tif')
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Splits
for split in ['train', 'val', 'test']:
    df = pd.DataFrame([[location, *list(np.random.randint(0, 2, size=(23,)))]])
    path = os.path.join('lc100_s3', f'multilabel-{split}.csv')
    df.to_csv(path, index=None)
    df = pd.DataFrame([[location, files[0]]])
    path = os.path.join('lc100_s3', f'static_fnames-{split}.csv')
    df.to_csv(path, header=None, index=None)

# Zip
shutil.make_archive('lc100_s3', 'zip', '.', 'lc100_s3')
