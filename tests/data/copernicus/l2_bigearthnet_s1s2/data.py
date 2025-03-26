#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'width': SIZE,
    'height': SIZE,
    'crs': CRS.from_wkt("""
PROJCS["WGS 84 / UTM zone 29N",
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
    PARAMETER["central_meridian",-9],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32629"]]
    """),
    'transform': Affine(10.0, 0.0, 692400.0, 0.0, -10.0, 5892840.0),
}

# Sentinel-1
s1_path = 'S1A_IW_GRDH_1SDV_20180420T063852/S1A_IW_GRDH_1SDV_20180420T063852_29UPU_77_6/S1A_IW_GRDH_1SDV_20180420T063852_29UPU_77_6_allbands.tif'
s1_directory = 'bigearthnet_s1s2/BigEarthNet-S1-5%'
os.makedirs(os.path.dirname(os.path.join(s1_directory, s1_path)), exist_ok=True)
profile['count'] = 2
profile['dtype'] = 'float32'
Z = np.random.rand(profile['height'], profile['width'])
path = os.path.join(s1_directory, s1_path)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Sentinel-2
s2_path = 'S2B_MSIL2A_20180421T114349_N9999_R123_T29UPU/S2B_MSIL2A_20180421T114349_N9999_R123_T29UPU_77_06/S2B_MSIL2A_20180421T114349_N9999_R123_T29UPU_77_06_allbands.tif'
s2_directory = 'bigearthnet_s1s2/BigEarthNet-S2-5%'
os.makedirs(os.path.dirname(os.path.join(s2_directory, s2_path)), exist_ok=True)
profile['count'] = 12
profile['dtype'] = 'uint16'
Z = np.random.randint(
    np.iinfo(profile['dtype']).min,
    np.iinfo(profile['dtype']).max,
    size=(profile['height'], profile['width']),
    dtype=profile['dtype'],
)
path = os.path.join(s2_directory, s2_path)
with rio.open(path, 'w', **profile) as src:
    for i in range(1, profile['count'] + 1):
        src.write(Z, i)

# Splits
df = pd.DataFrame([[s1_path, s2_path, *list(np.random.randint(0, 2, size=(19,)))]])
for split in ['train', 'val', 'test']:
    path = os.path.join('bigearthnet_s1s2', f'multilabel-{split}.csv')
    df.to_csv(path, index=False)

# Zip
shutil.make_archive('bigearthnetv2', 'zip', '.', 'bigearthnet_s1s2')
