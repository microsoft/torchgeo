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

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'height': SIZE,
    'width': SIZE,
    'count': 1,
    'crs': CRS.from_wkt("""
PROJCS["ETRS89-extended / LAEA Europe",
    GEOGCS["ETRS89",
        DATUM["European_Terrestrial_Reference_System_1989",
            SPHEROID["GRS 1980",6378137,298.257222101,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6258"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4258"]],
    PROJECTION["Lambert_Azimuthal_Equal_Area"],
    PARAMETER["latitude_of_center",52],
    PARAMETER["longitude_of_center",10],
    PARAMETER["false_easting",4321000],
    PARAMETER["false_northing",3210000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Northing",NORTH],
    AXIS["Easting",EAST],
    AUTHORITY["EPSG","3035"]]
    """),
    'transform': Affine(1113.2, 0.0, 3307317.2, 0.0, -1113.2, 3575598.4000000004),
}

Z = np.random.random(size=(profile['height'], profile['width']))
files = [
    '2021-01-01_2021-04-01.tif',
    '2021-04-01_2021-07-01.tif',
    '2021-07-01_2021-10-01.tif',
    '2021-10-01_2021-12-31.tif',
]
for variable in ['no2', 'os']:
    pid = f'EEA_1kmgrid_2021_{variable}_avg_34_13'

    # Image (annual)
    directory = os.path.join('airquality_s5p', variable, 's5p_annual', pid)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, files[-1])
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
