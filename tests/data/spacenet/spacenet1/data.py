#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 2

dataset_id = 'SN1_buildings'

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


def create_directories(base_path: str, band_counts: list[int]) -> None:
    for count in band_counts:
        os.makedirs(os.path.join(base_path, f'{count}band'), exist_ok=True)


def generate_geotiff_files(
    base_path: str, band_counts: list[int], profile: dict[str, Any], Z: np.ndarray
) -> None:
    for count in band_counts:
        for i in range(1, 5):
            path = os.path.join(
                base_path, f'{count}band', f'{count}band_AOI_1_RIO_img{i}.tif'
            )
            profile['count'] = count
            with rasterio.open(path, 'w', **profile) as src:
                for j in range(1, count + 1):
                    src.write(Z, j)


def generate_geojson_files(base_path: str, geojson: dict[str, Any]) -> None:
    os.makedirs(os.path.join(base_path, 'geojson'), exist_ok=True)
    for i in range(1, 4):
        path = os.path.join(base_path, 'geojson', f'Geo_AOI_1_RIO_img{i}.geojson')
        with open(path, 'w') as src:
            if i % 2 == 0:
                json.dump(geojson, src)


def compute_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Generate dummy GeoJSON files for building footprints
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

# Remove existing data if it exists
if os.path.exists(dataset_id):
    shutil.rmtree(dataset_id)

train_base_path = os.path.join(dataset_id, 'train')
test_base_path = os.path.join(dataset_id, 'test')

# Create directories and generate dummy GeoTIFF files for train dataset
create_directories(train_base_path, [3, 8])
generate_geotiff_files(train_base_path, [3, 8], profile, Z)
generate_geojson_files(train_base_path, geojson)

# Create directories and generate dummy GeoTIFF files for test dataset (only 3band and 8band)
create_directories(test_base_path, [3, 8])
generate_geotiff_files(test_base_path, [3, 8], profile, Z)

# Create tarballs for train and test datasets
shutil.make_archive(
    os.path.join(dataset_id, 'SN1_buildings_train_AOI_1_Rio'),
    'gztar',
    dataset_id,
    'train',
)
shutil.make_archive(
    os.path.join(dataset_id, 'SN1_buildings_test_AOI_1_Rio'),
    'gztar',
    dataset_id,
    'test',
)

# Compute and print MD5 checksums for the generated tarballs
print('MD5 Checksums for Train Dataset:')
train_tarball_path = os.path.join(dataset_id, 'SN1_buildings_train_AOI_1_Rio.tar.gz')
if os.path.exists(train_tarball_path):
    print(f'Train: {compute_md5(train_tarball_path)}')

print('\nMD5 Checksums for Test Dataset:')
test_tarball_path = os.path.join(dataset_id, 'SN1_buildings_test_AOI_1_Rio.tar.gz')
if os.path.exists(test_tarball_path):
    print(f'Test: {compute_md5(test_tarball_path)}')
