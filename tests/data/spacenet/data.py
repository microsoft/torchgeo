#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 2

dataset_id = 'SN1_buildings'
os.makedirs(os.path.join(dataset_id, 'tarballs'), exist_ok=True)

profile = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'width': SIZE,
    'height': SIZE,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(
        4.489235388119662e-06,
        0.0,
        -43.7732462563,
        0.0,
        -4.486127586210932e-06,
        -22.9214851954,
    ),
}

np.random.seed(0)
Z = np.random.randint(np.iinfo('uint8').max, size=(SIZE, SIZE), dtype='uint8')

for count in [3, 8]:
    os.makedirs(os.path.join(dataset_id, 'train', f'{count}band'), exist_ok=True)
    for i in range(1, 4):
        path = os.path.join(
            dataset_id, 'train', f'{count}band', f'3band_AOI_1_RIO_img{i}.tif'
        )
        profile['count'] = count
        with rasterio.open(path, 'w', **profile) as src:
            for j in range(1, count + 1):
                src.write(Z, j)

    shutil.make_archive(
        os.path.join(
            dataset_id, 'tarballs', f'SN1_buildings_train_AOI_1_Rio_{count}band'
        ),
        'gztar',
        os.path.join(dataset_id, 'train'),
        f'{count}band',
    )

geojson = {
    'type': 'FeatureCollection',
    'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
    'features': [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-43.7720361, -22.922229499999958, 0.0],
                        [-43.772064, -22.9222724, 0.0],
                        [-43.772102399999937, -22.922247399999947, 0.0],
                        [-43.772074499999974, -22.9222046, 0.0],
                        [-43.7720361, -22.922229499999958, 0.0],
                    ]
                ],
            },
        }
    ],
}

os.makedirs(os.path.join(dataset_id, 'train', 'geojson'), exist_ok=True)
for i in range(1, 3):
    path = os.path.join(dataset_id, 'train', 'geojson', f'Geo_AOI_1_RIO_img{i}.geojson')
    with open(path, 'w') as src:
        json.dump(geojson, src)

shutil.make_archive(
    os.path.join(
        dataset_id, 'tarballs', 'SN1_buildings_train_AOI_1_Rio_geojson_buildings'
    ),
    'gztar',
    os.path.join(dataset_id, 'train'),
    'geojson',
)
