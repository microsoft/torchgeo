#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import zipfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32
np.random.seed(0)

years = [1996, 2020]


def create_file(path: str) -> None:
    """Create a fake GMW GeoTIFF tile."""
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 1,
        'crs': CRS.from_epsg(4326),
        'transform': Affine(0.000223, 0.0, 100.0, 0.0, -0.000223, 1.0),
        'height': SIZE,
        'width': SIZE,
        'compress': 'lzw',
    }
    Z = np.random.randint(0, 2, size=(SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, 'w', **profile) as src:
        src.write(Z, 1)


if __name__ == '__main__':
    for year in years:
        tif_filename = os.path.join(
            'tests', 'data', 'globalmangrovewatch',
            f'gmw_v3_{year}_N00E100.tif',
        )
        create_file(tif_filename)

        zip_filename = os.path.join(
            'tests', 'data', 'globalmangrovewatch',
            f'gmw_v3_{year}_gtiff.zip',
        )
        with zipfile.ZipFile(zip_filename, 'w') as zf:
            zf.write(tif_filename, arcname=os.path.basename(tif_filename))