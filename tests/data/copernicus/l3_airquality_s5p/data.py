#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'height': SIZE,
    'width': SIZE,
    'count': 1,
    'crs': CRS.from_epsg(3035),
    'transform': Affine(1113.2, 0.0, 3307317.2, 0.0, -1113.2, 3575598.4000000004),
}

Z = np.random.random(size=(profile['height'], profile['width']))
files = [
    '2021-01-01_2021-04-01.tif',
    '2021-04-01_2021-07-01.tif',
    '2021-07-01_2021-10-01.tif',
    '2021-10-01_2021-12-31.tif',
]
for variable in ['no2', 'o3']:
    pid = f'EEA_1kmgrid_2021_{variable}_avg_34_13'

    # Image (annual)
    directory = os.path.join('airquality_s5p', variable, 's5p_annual', pid)
    os.makedirs(directory, exist_ok=True)
    file = '2021-01-01_2021-12-31.tif'
    path = os.path.join(directory, file)
    with rio.open(path, 'w', **profile) as src:
        src.write(Z, 1)

    # Images (seasonal)
    directory = os.path.join('airquality_s5p', variable, 's5p_seasonal', pid)
    os.makedirs(directory, exist_ok=True)
    for file in files:
        path = os.path.join(directory, file)
        with rio.open(path, 'w', **profile) as src:
            src.write(Z, 1)

    # Label (annual)
    directory = os.path.join('airquality_s5p', variable, 'label_annual')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{pid}.tif')
    with rio.open(path, 'w', **profile) as src:
        src.write(Z, 1)

    # Splits
    directory = os.path.join('airquality_s5p', variable)
    for split in ['train', 'val', 'test']:
        with open(os.path.join(directory, f'{split}.csv'), 'w') as f:
            f.write(f'{pid}\n')

# Zip
shutil.make_archive('airquality_s5p', 'zip', '.', 'airquality_s5p')
