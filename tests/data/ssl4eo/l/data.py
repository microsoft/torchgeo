#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36
CHUNK_SIZE = 2**12

np.random.seed(0)

FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]

filenames: FILENAME_HIERARCHY = {
    'ssl4eo_l_tm_toa': {
        '0000002': {
            'LT05_172034_20010526': ['all_bands.tif'],
            'LT05_172034_20020310': ['all_bands.tif'],
            'LT05_172034_20020902': ['all_bands.tif'],
            'LT05_172034_20021121': ['all_bands.tif'],
        },
        '0000005': {
            'LT05_223084_20010413': ['all_bands.tif'],
            'LT05_223084_20011225': ['all_bands.tif'],
            'LT05_223084_20020619': ['all_bands.tif'],
            'LT5_223084_20020923': ['all_bands.tif'],
        },
    },
    'ssl4eo_l_etm_toa': {
        '0000002': {
            'LE07_172034_20010526': ['all_bands.tif'],
            'LE07_172034_20020310': ['all_bands.tif'],
            'LE07_172034_20020902': ['all_bands.tif'],
            'LE07_172034_20021121': ['all_bands.tif'],
        },
        '0000005': {
            'LE07_223084_20010413': ['all_bands.tif'],
            'LE07_223084_20011225': ['all_bands.tif'],
            'LE07_223084_20020619': ['all_bands.tif'],
            'LE07_223084_20020923': ['all_bands.tif'],
        },
    },
    'ssl4eo_l_etm_sr': {
        '0000002': {
            'LE07_172034_20010526': ['all_bands.tif'],
            'LE07_172034_20020310': ['all_bands.tif'],
            'LE07_172034_20020902': ['all_bands.tif'],
            'LE07_172034_20021121': ['all_bands.tif'],
        },
        '0000005': {
            'LE07_223084_20010413': ['all_bands.tif'],
            'LE07_223084_20011225': ['all_bands.tif'],
            'LE07_223084_20020619': ['all_bands.tif'],
            'LE07_223084_20020923': ['all_bands.tif'],
        },
    },
    'ssl4eo_l_oli_tirs_toa': {
        '0000002': {
            'LC08_172034_20210306': ['all_bands.tif'],
            'LC08_172034_20210829': ['all_bands.tif'],
            'LC08_172034_20211203': ['all_bands.tif'],
            'LC08_172034_20220715': ['all_bands.tif'],
        },
        '0000005': {
            'LC08_223084_20210412': ['all_bands.tif'],
            'LC08_223084_20211005': ['all_bands.tif'],
            'LC08_223084_20220618': ['all_bands.tif'],
            'LC08_223084_20221211': ['all_bands.tif'],
        },
    },
    'ssl4eo_l_oli_sr': {
        '0000002': {
            'LC08_172034_20210306': ['all_bands.tif'],
            'LC08_172034_20210829': ['all_bands.tif'],
            'LC08_172034_20211203': ['all_bands.tif'],
            'LC08_172034_20220715': ['all_bands.tif'],
        },
        '0000005': {
            'LC08_223084_20210412': ['all_bands.tif'],
            'LC08_223084_20211005': ['all_bands.tif'],
            'LC08_223084_20220618': ['all_bands.tif'],
            'LC08_223084_20221211': ['all_bands.tif'],
        },
    },
}

num_bands = {
    'ssl4eo_l_tm_toa': 7,
    'ssl4eo_l_etm_toa': 9,
    'ssl4eo_l_etm_sr': 6,
    'ssl4eo_l_oli_tirs_toa': 11,
    'ssl4eo_l_oli_sr': 7,
}


def create_file(path: str) -> None:
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': SIZE,
        'height': SIZE,
        'count': num_bands[path.split(os.sep)[1]],
        'crs': CRS.from_epsg(4326),
        'transform': Affine(
            0.00033331040066238285,
            0.0,
            40.31409193350423,
            0.0,
            -0.0002658855613264443,
            37.60408425220701,
        ),
        'compress': 'lzw',
        'predictor': 2,
    }

    Z = np.random.randn(SIZE, SIZE).astype(profile['dtype'])

    with rasterio.open(path, 'w', **profile) as src:
        for i in src.indexes:
            src.write(Z, i)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path)


if __name__ == '__main__':
    create_directory('.', filenames)

    directories = filenames.keys()
    for directory in directories:
        # Create tarball
        shutil.make_archive(directory, 'gztar', '.', directory)

        # Split tarball
        path = f'{directory}.tar.gz'
        paths = []
        with open(path, 'rb') as f:
            suffix = 'a'
            while chunk := f.read(CHUNK_SIZE):
                split = f'{path}a{suffix}'
                with open(split, 'wb') as g:
                    g.write(chunk)
                suffix = chr(ord(suffix) + 1)
                paths.append(split)

        os.remove(path)

        # Compute checksums
        for path in paths:
            with open(path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
                print(path, md5)
