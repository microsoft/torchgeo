#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 16

np.random.seed(0)


def create_file(path: str, dtype: str) -> None:
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'width': SIZE,
        'height': SIZE,
        'count': 1,
        'crs': CRS.from_epsg(32616),
        'transform': Affine(30.0, 0.0, 229800.0, 0.0, -30.0, 4585230.0),
        'compress': 'lzw',
        'predictor': 2,
    }

    Z = np.random.randint(size=(SIZE, SIZE), low=0, high=np.iinfo(dtype).max)

    with rasterio.open(path, 'w', **profile) as src:
        src.write(Z, 1)


bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'QA_AEROSOL']

os.makedirs(os.path.join('preprocessed', 'cdl'), exist_ok=True)
os.makedirs(os.path.join('preprocessed', 'landsat'), exist_ok=True)

create_file(os.path.join('preprocessed', 'cdl', '2023_30m_cdls.tif'), 'uint8')
for band in bands:
    path = f'LC09_L2SP_023032_20230620_20230622_02_T1_SR_{band}.TIF'
    path = os.path.join('preprocessed', 'landsat', path)
    create_file(path, 'uint16')

# Compress data
shutil.make_archive('preprocessed', 'gztar', '.', 'preprocessed')

# Compute checksums
with open('preprocessed.tar.gz', 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(md5)
