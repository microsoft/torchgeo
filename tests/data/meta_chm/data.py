#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

SIZE = 32
RES = 1.1943285669558463
TILE = SIZE * RES

COG_DIR = os.path.join('forests', 'v2', 'global', 'dinov3_global_chm_v2_ml3', 'chm')

np.random.seed(0)


def create_tile(quadkey: str, minx: float, miny: float, date: str) -> dict:
    maxx, maxy = minx + TILE, miny + TILE
    path = os.path.join(COG_DIR, f'{quadkey}.tif')
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 1,
        'crs': 'EPSG:3857',
        'transform': from_bounds(minx, miny, maxx, maxy, SIZE, SIZE),
        'height': SIZE,
        'width': SIZE,
        'compress': 'lzw',
    }
    data = np.random.randint(0, 30, size=(1, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)

    href = os.path.abspath(path)
    return {
        'datetime': pd.Timestamp(date, tz='UTC'),
        'assets': {'chm': {'href': href}},
        '_box_3857': box(minx, miny, maxx, maxy),
    }


if __name__ == '__main__':
    if os.path.exists(COG_DIR.split(os.sep)[0]):
        shutil.rmtree(COG_DIR.split(os.sep)[0])
    os.makedirs(COG_DIR)

    minx0, miny0 = 1490000.0, 6890000.0
    rows = [
        create_tile('1202102331', minx0, miny0, '2019-08-09'),
        create_tile('1202102333', minx0 + TILE, miny0, '2019-07-15'),
    ]

    gdf = gpd.GeoDataFrame(
        rows, geometry=[r.pop('_box_3857') for r in rows], crs='EPSG:3857'
    ).to_crs('EPSG:4326')
    gdf.to_parquet('items.parquet')
