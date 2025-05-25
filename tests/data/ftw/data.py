#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import zipfile

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

np.random.seed(0)

country = 'austria'
SIZE = 32
num_samples = {'train': 2, 'val': 2, 'test': 2}
BASE_PROFILE = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 4,
    'crs': 'EPSG:4326',
    'transform': Affine(5.4e-05, 0.0, 0, 0.0, 5.4e-05, 0),
    'blockxsize': SIZE,
    'blockysize': SIZE,
    'tiled': True,
    'interleave': 'pixel',
}


def create_image(fn: str) -> None:
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    profile = BASE_PROFILE.copy()

    data = np.random.randint(0, 20000, size=(4, SIZE, SIZE), dtype=np.uint16)
    with rasterio.open(fn, 'w', **profile) as dst:
        dst.write(data)


def create_mask(fn: str, min_val: int, max_val: int) -> None:
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    profile = BASE_PROFILE.copy()
    profile['dtype'] = 'uint8'
    profile['nodata'] = 0
    profile['count'] = 1

    data = np.random.randint(min_val, max_val, size=(1, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(fn, 'w', **profile) as dst:
        dst.write(data)


if __name__ == '__main__':
    i = 0
    cols = {'aoi_id': [], 'split': []}
    for split, n in num_samples.items():
        for j in range(n):
            aoi = f'g_{i}'
            cols['aoi_id'].append(aoi)
            cols['split'].append(split)

            create_image(os.path.join(country, 's2_images', 'window_a', f'{aoi}.tif'))
            create_image(os.path.join(country, 's2_images', 'window_b', f'{aoi}.tif'))

            create_mask(
                os.path.join(country, 'label_masks', 'semantic_2class', f'{aoi}.tif'),
                0,
                1,
            )
            create_mask(
                os.path.join(country, 'label_masks', 'semantic_3class', f'{aoi}.tif'),
                0,
                2,
            )
            create_mask(
                os.path.join(country, 'label_masks', 'instance', f'{aoi}.tif'), 0, 100
            )

            i += 1

    # Create an extra train file to test for missing other files
    aoi = f'g_{i}'
    cols['aoi_id'].append(aoi)
    cols['split'].append(split)
    create_image(os.path.join(country, 's2_images', 'window_a', f'{aoi}.tif'))

    # Write parquet index
    df = pd.DataFrame(cols)
    df.to_parquet(os.path.join(country, f'chips_{country}.parquet'))

    # archive to zip
    with zipfile.ZipFile(f'{country}.zip', 'w') as zipf:
        for root, _, files in os.walk(country):
            for file in files:
                output_fn = os.path.join(root, file)
                zipf.write(output_fn, os.path.relpath(output_fn, country))

    # Compute checksums
    with open(f'{country}.zip', 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{md5}')
