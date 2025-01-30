#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
DTYPE = 'int16'

np.random.seed(0)

wkt = """
PROJCS["WGS 84 / UTM zone 40N",
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
    PARAMETER["central_meridian",57],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32640"]]
"""

profile = {
    'driver': 'GTiff',
    'dtype': DTYPE,
    'nodata': -32768.0,
    'width': SIZE,
    'height': SIZE,
    'count': 224,
    'crs': CRS.from_wkt(wkt),
    'transform': Affine(30.0, 0.0, 283455.0, 0.0, -30.0, 2786715.0),
}

filename = 'ENMAP01-____L2A-DT0000001053_20220611T072305Z_002_V010400_20231221T134421Z-SPECTRAL_IMAGE_COG.tiff'

Z = np.random.randint(
    np.iinfo(DTYPE).min, np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE
)
with rasterio.open(filename, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)
