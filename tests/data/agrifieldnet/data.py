#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def generate_test_data(paths: str) -> str:
    """Create test data archive for AgriFieldNet dataset.

    Args:
        paths: path to store test data
        n_samples: number of samples.

    Returns:
        md5 hash of created archive
    """
    dtype = np.uint8
    dtype_max = np.iinfo(dtype).max

    SIZE = 32

    np.random.seed(0)

    bands = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B11',
        'B12',
    )

    profile = {
        'dtype': dtype,
        'width': SIZE,
        'height': SIZE,
        'count': 1,
        'crs': CRS.from_epsg(32644),
        'transform': Affine(10.0, 0.0, 535840.0, 0.0, -10.0, 3079680.0),
    }

    source_dir = os.path.join(paths, 'source')
    train_mask_dir = os.path.join(paths, 'train_labels')
    test_field_dir = os.path.join(paths, 'test_labels')

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_field_dir, exist_ok=True)

    source_unique_folder_ids = [
        '32407',
        '8641e',
        'a419f',
        'eac11',
        'ff450',
        '001c1',
        '004fa',
        '005fe',
        '00720',
        '00c23',
    ]
    train_folder_ids = source_unique_folder_ids[0:5]
    test_folder_ids = source_unique_folder_ids[3:5]

    for id in source_unique_folder_ids:
        directory = os.path.join(
            source_dir, 'ref_agrifieldnet_competition_v1_source_' + id
        )
        os.makedirs(directory, exist_ok=True)

        for band in bands:
            train_arr = np.random.randint(dtype_max, size=(SIZE, SIZE), dtype=dtype)
            path = os.path.join(
                directory, f'ref_agrifieldnet_competition_v1_source_{id}_{band}_10m.tif'
            )
            with rasterio.open(path, 'w', **profile) as src:
                src.write(train_arr, 1)

    for id in train_folder_ids:
        train_mask_arr = np.random.randint(size=(SIZE, SIZE), low=0, high=6)
        path = os.path.join(
            train_mask_dir, f'ref_agrifieldnet_competition_v1_labels_train_{id}.tif'
        )
        with rasterio.open(path, 'w', **profile) as src:
            src.write(train_mask_arr, 1)

        train_field_arr = np.random.randint(20, size=(SIZE, SIZE), dtype=np.uint16)
        path = os.path.join(
            train_mask_dir,
            f'ref_agrifieldnet_competition_v1_labels_train_{id}_field_ids.tif',
        )
        with rasterio.open(path, 'w', **profile) as src:
            src.write(train_field_arr, 1)

    for id in test_folder_ids:
        test_field_arr = np.random.randint(10, 30, size=(SIZE, SIZE), dtype=np.uint16)
        path = os.path.join(
            test_field_dir,
            f'ref_agrifieldnet_competition_v1_labels_test_{id}_field_ids.tif',
        )
        with rasterio.open(path, 'w', **profile) as src:
            src.write(test_field_arr, 1)


if __name__ == '__main__':
    generate_test_data(os.getcwd())
