#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
filename = 'Residential_2029.tif'

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'width': SIZE,
    'height': SIZE,
    'crs': CRS.from_wkt("""
PROJCS["WGS 84 / UTM zone 33N",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",15],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32633"]]
    """),
    'transform': Affine(9.894933409914302, 0.0, 453604.151188705,
       0.0, -9.786558657346177, 5165115.807450672),
}

# Sentinel-1
directory = os.path.join('eurosat_s1', 'all_imgs', 'Residential')
os.makedirs(directory, exist_ok=True)
profile['count'] = 2
profile['dtype'] = 'float32'
Z = np.random.rand(profile['height'], profile['width'])
path = os.path.join(directory, filename)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Sentinel-2
directory = os.path.join('eurosat_s2', 'all_imgs', 'Residential')
os.makedirs(directory, exist_ok=True)
profile['count'] = 13
profile['dtype'] = 'uint16'
Z = np.random.randint(
    np.iinfo(profile['dtype']).min,
    np.iinfo(profile['dtype']).max,
    size=(profile['height'], profile['width']), dtype=profile['dtype']
)
path = os.path.join(directory, filename)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

filename = filename.replace('.tif', '.jpg')
for directory in ['eurosat_s1', 'eurosat_s2']:
    # Splits
    for split in ['train', 'val', 'test']:
        with open(os.path.join(directory, f'eurosat-{split}.csv'), 'w') as csv:
            csv.write(f'{filename}\n')

    # Zip
    shutil.make_archive(directory, 'zip', '.', directory)
