#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32

np.random.seed(0)

metadata = {
    'e2eb30d2c5eb52669c5e6a9db8973835': {
        'path': '1111013/01/e2eb30d2c5eb52669c5e6a9db8973835',
        'info': {
            'datasets': {
                # Rank is sometimes int, sometimes str
                'MS1_IVV_1111013_01_20170504': {
                    'name': 'MS1_IVV_1111013_01_20170504',
                    'dtype': 'float32',
                    'ptype': 'MS',
                    'rank': '1',
                    'pname': 'IVV',
                },
                'MS1_IVH_1111013_01_20170504': {
                    'name': 'MS1_IVH_1111013_01_20170504',
                    'dtype': 'float32',
                    'ptype': 'MS',
                    'rank': '1',
                    'pname': 'IVH',
                },
                'SL1_IVV_1111013_01_20170422': {
                    'name': 'SL1_IVV_1111013_01_20170422',
                    'dtype': 'float32',
                    'ptype': 'SL',
                    'rank': 1,
                    'pname': 'IVV',
                },
                'SL1_IVH_1111013_01_20170422': {
                    'name': 'SL1_IVH_1111013_01_20170422',
                    'dtype': 'float32',
                    'ptype': 'SL',
                    'rank': 1,
                    'pname': 'IVH',
                },
                'SL2_IVV_1111013_01_20170410': {
                    'name': 'SL2_IVV_1111013_01_20170410',
                    'dtype': 'float32',
                    'ptype': 'SL',
                    'rank': 2,
                    'pname': 'IVV',
                },
                'SL2_IVH_1111013_01_20170410': {
                    'name': 'SL2_IVH_1111013_01_20170410',
                    'dtype': 'float32',
                    'ptype': 'SL',
                    'rank': 2,
                    'pname': 'IVH',
                },
                'MK0_MLU_1111013_01_20170504': {
                    'name': 'MK0_MLU_1111013_01_20170504',
                    'dtype': 'uint8',
                    'ptype': 'MK',
                    'rank': 0,
                    'pname': 'MLU',
                },
            }
        },
    }
}

profile = {
    'driver': 'GTiff',
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_wkt("""
PROJCS["WGS 84 / Pseudo-Mercator",
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
    PROJECTION["Mercator_1SP"],
    PARAMETER["central_meridian",0],
    PARAMETER["scale_factor",1],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"],
    AUTHORITY["EPSG","3857"]]
    """),
    'transform': Affine(10.0, 0.0, -10003405.0, 0.0, -10.0, 4730925.0),
}

# Images and masks
directory = 'flood_s1'
for key in metadata.keys():
    subdir = os.path.join(directory, 'data', metadata[key]['path'])
    os.makedirs(subdir, exist_ok=True)
    for value in metadata[key]['info']['datasets'].values():
        path = os.path.join(subdir, value['name'] + '.tif')
        profile['dtype'] = value['dtype']

        if value['dtype'] == 'float32':
            Z = np.random.random(size=(profile['height'], profile['width']))
        else:
            Z = np.random.randint(
                0, 3, size=(profile['height'], profile['width']), dtype=profile['dtype']
            )
        with rio.open(path, 'w', **profile) as src:
            src.write(Z, 1)

# Splits
for split in ['train', 'val', 'test']:
    with open(os.path.join(directory, f'grid_dict_{split}.json'), 'w') as f:
        json.dump(metadata, f)

# Zip
shutil.make_archive(directory, 'zip', '.', directory)
