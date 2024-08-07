#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 128  # image width/height

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


values = [
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    41,
    42,
    51,
    52,
    53,
    54,
    55,
    56,
    62,
    63,
    64,
    65,
    72,
    73,
    74,
    75,
    83,
    84,
    85,
    91,
    92,
    93,
    94,
    95,
    127,
]

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

for state in ['dc', 'de', 'md', 'ny', 'pa', 'va', 'wv']:
    filename = f'{state}_lulc_2018_2022-Edition'

    # Create raster file
    with rasterio.open(f'{filename}.tif', 'w', **meta) as f:
        data = np.random.choice(values, size=(SIZE, SIZE))
        f.write(data, 1)

    # Compress file
    shutil.make_archive(filename, 'zip', '.', filename + '.tif')

    # Compute checksums
    with open(f'{filename}.zip', 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(state, repr(md5))
