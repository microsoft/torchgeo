#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
DTYPE = 'int16'

np.random.seed(0)

tiles = ['ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z']
patches = ['Y01460273_X05670694', 'Y01460273_X06950822']

profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'nodata': -32768.0,
    'width': SIZE,
    'height': SIZE,
    'count': 224,
    'crs': CRS.from_epsg(32618),
    'transform': Affine(30.0, 0.0, 691845.0, 0.0, -30.0, 4561935.0),
    'blockysize': 3,
    'tiled': False,
    'compress': 'deflate',
    'interleave': 'band',
}

root = 'hyspecnet-11k'
path = os.path.join(root, 'splits', 'easy')
os.makedirs(path, exist_ok=True)
for tile in tiles:
    for patch in patches:
        # Split CSV
        path = os.path.join(tile, f'{tile}-{patch}', f'{tile}-{patch}-DATA.npy')
        for split in ['train', 'val', 'test']:
            with open(os.path.join(root, 'splits', 'easy', f'{split}.csv'), 'a+') as f:
                f.write(f'{path}\n')

        # Spectral image
        path = os.path.join(root, 'patches', path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = path.replace('DATA.npy', 'SPECTRAL_IMAGE.TIF')
        Z = np.random.randint(
            np.iinfo(DTYPE).min, np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE
        )
        with rasterio.open(path, 'w', **profile) as src:
            for i in range(1, profile['count'] + 1):
                src.write(Z, i)

shutil.make_archive(f'{root}-01', 'gztar', '.', os.path.join(root, 'patches'))
shutil.make_archive(f'{root}-splits', 'gztar', '.', os.path.join(root, 'splits'))
