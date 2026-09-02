# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import os

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 8

np.random.seed(0)

# One small synthetic GeoTIFF per band, matching the real export naming
# convention: hansen_<band>_<region>.tif
BANDS = ['treecover2000', 'loss', 'lossyear', 'gain', 'datamask']
REGION = 'test'

profile = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'count': 1,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.0),
    'height': SIZE,
    'width': SIZE,
}

for band in BANDS:
    filename = f'hansen_{band}_{REGION}.tif'
    if band == 'lossyear':
        data = np.random.randint(0, 25, size=(SIZE, SIZE), dtype=np.uint8)
    elif band == 'treecover2000':
        data = np.random.randint(0, 101, size=(SIZE, SIZE), dtype=np.uint8)
    else:
        data = np.random.randint(0, 2, size=(SIZE, SIZE), dtype=np.uint8)

    with rasterio.open(filename, 'w', **profile) as f:
        f.write(data, 1)

print(f'Generated {len(BANDS)} fake GeoTIFFs in {os.getcwd()}')
