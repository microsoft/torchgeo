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

dataset_id = 'SN6_buildings'

profile = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'width': SIZE,
    'height': SIZE,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(
        4.489235388119662e-06, 0.0, 4.47917, 0.0, -4.486127586210932e-06, 51.9225
    ),
}

np.random.seed(0)
Z = np.random.randint(np.iinfo('uint8').max, size=(SIZE, SIZE), dtype='uint8')

# Define the types of imagery for SpaceNet6
imagery_types = ['PAN', 'PS-RGB', 'PS-RGBNIR', 'RGBNIR', 'SAR-Intensity']
imagery_channels = {
    'PAN': 1,
    'PS-RGB': 3,
    'PS-RGBNIR': 4,
    'RGBNIR': 4,
    'SAR-Intensity': 4,
}


def create_directories(base_path: str, imagery_types: list[str]) -> None:
    for imagery_type in imagery_types:
        os.makedirs(os.path.join(base_path, imagery_type), exist_ok=True)


def generate_geotiff_files(
    base_path: str,
    imagery_types: str,
    imagery_channels: int,
    profile: dict[str, Any],
    Z: np.ndarray,
    test: bool = False,
) -> None:
    for imagery_type in imagery_types:
        for i in range(1, 5):
            if test and imagery_type == 'SAR-Intensity':
                path = os.path.join(
                    base_path,
                    f'SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_20190804111224_20190804111453_tile_{i}.tif',
                )
            else:
                path = os.path.join(
                    base_path,
                    imagery_type,
                    f'SN6_Train_AOI_11_Rotterdam_{imagery_type}_20190804111224_20190804111453_tile_{i}.tif',
                )
            profile['count'] = imagery_channels[imagery_type]
            with rasterio.open(path, 'w', **profile) as src:
                for j in range(1, profile['count'] + 1):
                    src.write(Z, j)


def generate_geojson_files(base_path: str, geojson: dict[str, Any]) -> None:
    os.makedirs(os.path.join(base_path, 'geojson_buildings'), exist_ok=True)
    for i in range(1, 4):
        path = os.path.join(
            base_path,
            'geojson_buildings',
            f'SN6_Train_AOI_11_Rotterdam_Buildings_20190804111224_20190804111453_tile_{i}.geojson',
        )
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
                        [4.47917, 51.9225, 0.0],
                        [4.47920, 51.92255, 0.0],
                        [4.47925, 51.92252, 0.0],
                        [4.47922, 51.92247, 0.0],
                        [4.47917, 51.9225, 0.0],
                    ]
                ],
            },
        }
    ],
}

# Remove existing data if it exists
if os.path.exists(dataset_id):
    shutil.rmtree(dataset_id)

train_base_path = os.path.join(dataset_id, 'train/train/AOI_11_Rotterdam')
test_base_path = os.path.join(
    dataset_id, 'test/test_public/AOI_11_Rotterdam/SAR-Intensity'
)

# Create directories and generate dummy GeoTIFF files for train dataset
create_directories(train_base_path, imagery_types)
generate_geotiff_files(train_base_path, imagery_types, imagery_channels, profile, Z)
generate_geojson_files(train_base_path, geojson)

# Create directories and generate dummy GeoTIFF files for test dataset (only SAR-Intensity)
os.makedirs(test_base_path, exist_ok=True)
generate_geotiff_files(
    test_base_path, ['SAR-Intensity'], imagery_channels, profile, Z, test=True
)

# Create tarballs for train and test datasets
shutil.make_archive(
    os.path.join(dataset_id, 'SN6_buildings_AOI_11_Rotterdam_train'),
    'gztar',
    dataset_id,
    'train',
)
shutil.make_archive(
    os.path.join(dataset_id, 'SN6_buildings_AOI_11_Rotterdam_test'),
    'gztar',
    dataset_id,
    'test',
)

# Compute and print MD5 checksums for the generated tarballs
print('MD5 Checksums for Train Dataset:')
train_tarball_path = os.path.join(
    dataset_id, 'SN6_buildings_AOI_11_Rotterdam_train.tar.gz'
)
if os.path.exists(train_tarball_path):
    print(f'Train: {compute_md5(train_tarball_path)}')

print('\nMD5 Checksums for Test Dataset:')
test_tarball_path = os.path.join(
    dataset_id, 'SN6_buildings_AOI_11_Rotterdam_test.tar.gz'
)
if os.path.exists(test_tarball_path):
    print(f'Test: {compute_md5(test_tarball_path)}')
