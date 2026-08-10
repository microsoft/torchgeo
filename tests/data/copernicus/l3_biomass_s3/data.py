#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

MASK_SIZE = 32
IMAGE_SIZE = 8

np.random.seed(0)

LOCATIONS = [
    {
        'name': 'S30E140_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv4.0_02_11',
        'mask_hw': (MASK_SIZE, MASK_SIZE),
        'files': [
            ('S3A_20200319T233546_20200319T233846.tif', (IMAGE_SIZE, IMAGE_SIZE)),
            (
                'S3B_20200514T234457_20200514T234757.tif',
                (IMAGE_SIZE + 2, IMAGE_SIZE + 2),
            ),
        ],
        'static_file': 'S3B_20200514T234457_20200514T234757.tif',
    },
    {
        'name': 'S32E141_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv4.0_01_05',
        'mask_hw': (MASK_SIZE - 4, MASK_SIZE),
        'files': [
            (
                'S3A_20210119T033546_20210119T033846.tif',
                (IMAGE_SIZE - 1, IMAGE_SIZE + 1),
            ),
            ('S3B_20210314T034457_20210314T034757.tif', (IMAGE_SIZE + 1, IMAGE_SIZE)),
        ],
        'static_file': 'S3B_20210314T034457_20210314T034757.tif',
    },
]

profile = {'driver': 'GTiff', 'crs': CRS.from_epsg(4326)}

# Images and masks
image_profile = profile | {
    'transform': Affine(
        0.0026949458523585646,
        0.0,
        142.50604172686855,
        0.0,
        -0.0026949458523585646,
        -30.24807224687253,
    ),
    'count': 21,
    'dtype': 'float32',
    'nodata': -np.inf,
}

mask_profile = profile | {
    'transform': Affine(
        0.00088888888888,
        0.0,
        142.5066666666416,
        0.0,
        -0.00088888888888,
        -30.25066666666416,
    ),
    'count': 1,
    'dtype': 'uint16',
}

biomass_dir = os.path.join('biomass_s3', 'biomass')
os.makedirs(biomass_dir, exist_ok=True)

static_entries: list[tuple[str, str]] = []

for location in LOCATIONS:
    directory = os.path.join('biomass_s3', 's3_olci', location['name'])
    os.makedirs(directory, exist_ok=True)

    for fname, (height, width) in location['files']:
        profile_args = image_profile | {'height': height, 'width': width}
        data = np.random.random(size=(height, width))
        if fname == 'S3B_20200514T234457_20200514T234757.tif':
            data[0, 0] = -np.inf
        path = os.path.join(directory, fname)
        with rio.open(path, 'w', **profile_args) as src:
            for band in range(1, profile_args['count'] + 1):
                src.write(data, band)

    mask_h, mask_w = location['mask_hw']
    mask_args = mask_profile | {'height': mask_h, 'width': mask_w}
    mask_data = np.random.randint(100, size=(mask_h, mask_w), dtype=np.uint16)
    mask_path = os.path.join(biomass_dir, f'{location["name"]}.tif')
    with rio.open(mask_path, 'w', **mask_args) as src:
        src.write(mask_data, 1)

    static_entries.append((location['name'], location['static_file']))

# Splits
directory = 'biomass_s3'
for split in ['train', 'val', 'test']:
    with open(os.path.join(directory, f'static_fnames-{split}.csv'), 'w') as f:
        for pid, fname in static_entries:
            f.write(f'{pid},{fname}\n')

# Zip
shutil.make_archive(directory, 'zip', '.', directory)
