#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import tarfile

import numpy as np
import rasterio
from affine import Affine

np.random.seed(0)

SIZE = 32
BASE_PROFILE = {
    'blockxsize': 224,
    'blockysize': 4,
    'count': 4,
    'crs': 'EPSG:4326',
    'driver': 'GTiff',
    'dtype': 'uint16',
    'height': SIZE,
    'width': SIZE,
    'interleave': 'pixel',
    'nodata': 65535.0,
    'tiled': False,
    'transform': Affine(2.2e-05, 0.0, 31.084, 0.0, -2.2e-05, -22.08),
}
FILENAMES = (
    'CM0112710_620_2017-02.tif',
    'CI0610961_12784_2020-12.tif',
    'CI0081173_31736_2021-12.tif',
    'ET2003128_36839_2023-04.tif',
    'ET2497199_2023-04.tif',
    'KE1131804_15075_2021-05.tif',
    'ZW0447510_8193_2020-07.tif',
    'ZW1280035_37945_2023-06.tif',
)
IMAGE_DIR = 'images'
MASK_DIR = 'labels'


def create_image(fn: str) -> None:
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    profile = BASE_PROFILE.copy()

    data = np.random.randint(0, 20000, size=(4, SIZE, SIZE), dtype=np.uint16)
    with rasterio.open(fn, 'w', **profile) as dst:
        dst.write(data)


def create_mask(fn: str, min_val: int, max_val: int) -> None:
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    profile = BASE_PROFILE.copy()
    profile['interleave'] = 'band'
    profile['dtype'] = 'uint8'
    profile['nodata'] = None
    profile['count'] = 1

    data = np.random.randint(min_val, max_val, size=(1, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(fn, 'w', **profile) as dst:
        dst.write(data)


if __name__ == '__main__':
    for filename in FILENAMES:
        create_image(os.path.join(IMAGE_DIR, filename))
        create_mask(os.path.join(MASK_DIR, filename), 0, 2)

    # Combine images and masks directory into tar.gz archive
    output_file = 'lacuna-field-boundaries.tar.gz'
    with tarfile.open(output_file, 'w:gz') as tar:
        tar.add(IMAGE_DIR)
        tar.add(MASK_DIR)

    # Compute checksums
    with open(output_file, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{md5}')
