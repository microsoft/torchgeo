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

# Real tiles cover 1 degree with 4,500 x 4,500 pixels (~25 m)
RES = 0.000222222222222

np.random.seed(0)

years = [1996, 2020]


def create_file(path: str) -> None:
    """Create a fake GMW GeoTIFF tile."""
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0,
        'count': 1,
        'crs': CRS.from_epsg(4326),
        # Upper left corner of the N00E008 tile
        'transform': Affine(RES, 0.0, 8.0, 0.0, -RES, 0.0),
        'height': SIZE,
        'width': SIZE,
        'compress': 'lzw',
    }
    Z = np.random.randint(0, 2, size=(SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, 'w', **profile) as src:
        src.write(Z, 1)


if __name__ == '__main__':
    root = os.path.join('tests', 'data', 'globalmangrovewatch')
    for year in years:
        directory = os.path.join(root, f'gmw_v3_{year}')
        os.makedirs(directory, exist_ok=True)

        tif_filename = os.path.join(directory, f'GMW_N00E008_{year}_v3.tif')
        create_file(tif_filename)

        zip_filename = os.path.join(root, f'gmw_v3_{year}_gtiff.zip')
        with zipfile.ZipFile(zip_filename, 'w') as zf:
            zf.write(
                tif_filename,
                arcname=os.path.join(f'gmw_v3_{year}', f'GMW_N00E008_{year}_v3.tif'),
            )
