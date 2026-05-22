# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.control import GroundControlPoint

EPSG = 4326
SIZE = 16


def write_raster_no_affine(path: str | None = None, has_gcps: bool = True) -> None:
    """Write a raster with GCPs but no affine transform.

    Args:
        epsg: EPSG of file.
        path: Output file path.
    """
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 1,
        'width': SIZE,
        'height': SIZE,
        'transform': Affine.identity(),
        'crs': None,
        'nodata': 0,
    }

    if path is None:
        name = f'gcps_{str(has_gcps).lower()}'
        path = os.path.join(name, f'{name}.tif')

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    with rio.open(path, 'w', **profile) as ds:
        data = np.ones((1, SIZE, SIZE), dtype=np.uint8)
        ds.write(data)

        if has_gcps:
            # Four corner GCPs
            gcps = [
                GroundControlPoint(row=0, col=0, x=0, y=0),
                GroundControlPoint(row=0, col=SIZE - 1, x=10, y=0),
                GroundControlPoint(row=SIZE - 1, col=SIZE - 1, x=10, y=10),
                GroundControlPoint(row=SIZE - 1, col=0, x=0, y=10),
            ]

            ds.gcps = (gcps, rio.crs.CRS.from_epsg(EPSG))


if __name__ == '__main__':
    for has_gcps in [True, False]:
        write_raster_no_affine(has_gcps=has_gcps)
