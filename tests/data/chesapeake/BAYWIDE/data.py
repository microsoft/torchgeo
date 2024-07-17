#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import subprocess

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 128  # image width/height
NUM_CLASSES = 14

np.random.seed(0)

filename = 'Baywide_13Class_20132014'
wkt = """
PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",
    GEOGCS["NAD83",
        DATUM["North_American_Datum_1983",
            SPHEROID["GRS 1980",6378137,298.257222101004,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6269"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4269"]],
    PROJECTION["Albers_Conic_Equal_Area"],
    PARAMETER["latitude_of_center",23],
    PARAMETER["longitude_of_center",-96],
    PARAMETER["standard_parallel_1",29.5],
    PARAMETER["standard_parallel_2",45.5],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""
cmap = {
    0: (0, 0, 0, 255),
    1: (0, 197, 255, 255),
    2: (0, 168, 132, 255),
    3: (38, 115, 0, 255),
    4: (76, 230, 0, 255),
    5: (163, 255, 115, 255),
    6: (255, 170, 0, 255),
    7: (255, 0, 0, 255),
    8: (156, 156, 156, 255),
    9: (0, 0, 0, 255),
    10: (115, 115, 0, 255),
    11: (230, 230, 0, 255),
    12: (255, 255, 115, 255),
    13: (197, 0, 255, 255),
}


meta = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_wkt(wkt),
    'transform': Affine(1.0, 0.0, 1303555.0000000005, 0.0, -1.0, 2535064.999999998),
}

# Remove old data
if os.path.exists(f'{filename}.tif'):
    os.remove(f'{filename}.tif')

# Create raster file
with rasterio.open(f'{filename}.tif', 'w', **meta) as f:
    data = np.random.randint(NUM_CLASSES, size=(SIZE, SIZE), dtype=np.uint8)
    f.write(data, 1)
    f.write_colormap(1, cmap)

# Create zip file
# 7z required to create a zip file using the proprietary DEFLATE64 compression algorithm
# https://github.com/brianhelba/zipfile-deflate64/issues/19#issuecomment-1006077294
subprocess.run(
    ['7z', 'a', f'{filename}.zip', '-mm=DEFLATE64', f'{filename}.tif'],
    capture_output=True,
    check=True,
)

# Compute checksums
with open(f'{filename}.zip', 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(repr(md5))
