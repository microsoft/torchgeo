# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import zipfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def generate_test_data(paths: str) -> None:
    """Create test data for ChesapeakeCVPR dataset.

    Args:
        paths: path to store test data
    """
    dtype = np.uint8
    dtype_max = np.iinfo(dtype).max

    SIZE = 32

    np.random.seed(0)

    states = ['de', 'md']
    splits = ['train', 'test']
    images = ['naip-new', 'naip-old']
    masks = ['lc', 'buildings']

    directory = os.path.join(paths, 'cvpr_chesapeake_landcover')
    os.makedirs(directory, exist_ok=True)

    for state in states:
        crs = CRS.from_epsg(26918) if state == 'de' else CRS.from_epsg(26917)
        x_start = 269365.7586466661 if 'de' in state else 784488.0
        for split in splits:
            folder_names = f'{state}_1m_2013_extended-debuffered-{split}_tiles'
            x_coord = x_start + SIZE if 'test' in split else x_start
            transform = (
                Affine(1, 0.0, x_coord, 0.0, -1, 4387676.633364204)
                if 'de' in state
                else Affine(1, 0.0, x_coord, 0.0, -1, 4389476.0)
            )

            profile = {
                'dtype': dtype,
                'width': SIZE,
                'height': SIZE,
                'count': 1,
                'crs': crs,
                'transform': transform,
            }
            folder = os.path.join(directory, folder_names)
            os.makedirs(folder, exist_ok=True)

            for image in images:
                tiles_names = (
                    f'm_3807504_ne_18_1_{image}.tif'
                    if state == 'de'
                    else f'm_3907822_nw_17_1_{image}.tif'
                )
                count = 4
                profile['count'] = count
                arr = np.random.randint(
                    dtype_max, size=(count, SIZE, SIZE), dtype=dtype
                )
                path = os.path.join(folder, tiles_names)
                with rasterio.open(path, 'w', **profile) as src:
                    src.write(arr)
            for mask in masks:
                tiles_names = (
                    f'm_3807504_ne_18_1_{mask}.tif'
                    if state == 'de'
                    else f'm_3907822_nw_17_1_{mask}.tif'
                )
                profile['count'] = 1

                arr = np.random.randint(
                    4 if 'lc' in mask else 2, size=(1, SIZE, SIZE), dtype=dtype
                )
                path = os.path.join(folder, tiles_names)
                with rasterio.open(path, 'w', **profile) as src:
                    src.write(arr)
            # extension layer
            extension_dir = os.path.join(
                paths, 'cvpr_chesapeake_landcover_prior_extension'
            )
            extension_folder = os.path.join(extension_dir, folder_names)
            os.makedirs(extension_folder, exist_ok=True)
            extension_name = (
                'm_3807504_ne_18_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif'
                if state == 'de'
                else 'm_3907822_nw_17_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif'
            )
            profile['count'] = 4
            extension_arr = np.random.randint(
                dtype_max, size=(4, SIZE, SIZE), dtype=dtype
            )
            path = os.path.join(extension_folder, extension_name)
            with rasterio.open(path, 'w', **profile) as src:
                src.write(extension_arr)

    directories = [directory, extension_dir]
    for d in directories:
        with zipfile.ZipFile(f'{d}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(d):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, d))


if __name__ == '__main__':
    generate_test_data(os.getcwd())
