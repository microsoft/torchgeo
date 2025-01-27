#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import zstandard as zstd
import tarfile

# Constants
IMG_SIZE = 120
ROOT_DIR = '.'

# Sample patch definitions
SAMPLE_PATCHES = [
    {
        's2_name': 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57',
        's2_base': 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP',
        's1_name': 'S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39',
        's1_base': 'S1A_IW_GRDH_1SDV_20170613T165043',
        'split': 'train',
    },
    {
        's2_name': 'S2B_MSIL2A_20170615T102019_N9999_R122_T32TNS_45_23',
        's2_base': 'S2B_MSIL2A_20170615T102019_N9999_R122_T32TNS',
        's1_name': 'S1A_IW_GRDH_1SDV_20170615T170156_32TNS_77_12',
        's1_base': 'S1A_IW_GRDH_1SDV_20170615T170156',
        'split': 'val',
    },
    {
        's2_name': 'S2A_MSIL2A_20170618T101021_N9999_R022_T32TQR_89_34',
        's2_base': 'S2A_MSIL2A_20170618T101021_N9999_R022_T32TQR',
        's1_name': 'S1A_IW_GRDH_1SDV_20170618T165722_32TQR_92_45',
        's1_base': 'S1A_IW_GRDH_1SDV_20170618T165722',
        'split': 'test',
    },
]

S1_BANDS = ['VV', 'VH']
S2_BANDS = [
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
]


def create_directory_structure() -> None:
    """Create the base directory structure"""

    for dir_name in ['BigEarthNet-S1', 'BigEarthNet-S2', 'Reference_Maps']:
        if os.path.exists(os.path.join(ROOT_DIR, dir_name)):
            shutil.rmtree(os.path.join(ROOT_DIR, dir_name))
        Path(os.path.join(ROOT_DIR, dir_name)).mkdir(parents=True, exist_ok=True)


def create_dummy_image(path: str, shape: tuple[int, int], dtype: str) -> None:
    """Create a dummy GeoTIFF file"""
    if dtype == 's1':
        data = np.random.randint(-25, 0, shape).astype(np.int16)
    elif dtype == 's2':
        data = np.random.randint(0, 10000, shape).astype(np.int16)
    else:  # reference map
        data = np.random.randint(0, 19, shape).astype(np.uint8)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs',
        transform=rasterio.transform.from_origin(0, 0, 10, 10),
    ) as dst:
        dst.write(data, 1)


def generate_sample(patch_info: dict) -> None:
    """Generate a complete sample with S1, S2 and reference data"""
    # Create S1 data
    s1_dir = os.path.join(
        ROOT_DIR, 'BigEarthNet-S1', patch_info['s1_base'], patch_info['s1_name']
    )
    os.makedirs(s1_dir, exist_ok=True)

    for band in S1_BANDS:
        path = os.path.join(s1_dir, f'{patch_info["s1_name"]}_{band}.tif')
        create_dummy_image(path, (IMG_SIZE, IMG_SIZE), 's1')

    # Create S2 data
    s2_dir = os.path.join(
        ROOT_DIR, 'BigEarthNet-S2', patch_info['s2_base'], patch_info['s2_name']
    )
    os.makedirs(s2_dir, exist_ok=True)

    for band in S2_BANDS:
        path = os.path.join(s2_dir, f'{patch_info["s2_name"]}_{band}.tif')
        create_dummy_image(path, (IMG_SIZE, IMG_SIZE), 's2')

    # Create reference map
    ref_dir = os.path.join(
        ROOT_DIR, 'Reference_Maps', patch_info['s2_base'], patch_info['s2_name']
    )
    os.makedirs(ref_dir, exist_ok=True)

    path = os.path.join(ref_dir, f'{patch_info["s2_name"]}_reference_map.tif')
    create_dummy_image(path, (IMG_SIZE, IMG_SIZE), 'reference')


def create_metadata() -> None:
    """Create metadata parquet file"""
    records = []

    for patch in SAMPLE_PATCHES:
        records.append(
            {
                'patch_id': patch['s2_name'],
                's1_name': patch['s1_name'],
                'split': patch['split'],
                'labels': np.random.choice(range(19), size=3, replace=False).tolist(),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_parquet(os.path.join(ROOT_DIR, 'metadata.parquet'))


def compress_directory(dirname: str) -> None:
    """Compress directory using tar+zstd"""
    tar_path = os.path.join(ROOT_DIR, f'{dirname}.tar')
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(os.path.join(ROOT_DIR, dirname), arcname=dirname)

    with open(tar_path, 'rb') as f_in:
        data = f_in.read()
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(data)
        with open(f'{tar_path}.zst', 'wb') as f_out:
            f_out.write(compressed)

    os.remove(tar_path)


def main() -> None:
    # Create directories and generate data
    create_directory_structure()

    for patch_info in SAMPLE_PATCHES:
        generate_sample(patch_info)

    create_metadata()

    # Compress directories
    for dirname in ['BigEarthNet-S1', 'BigEarthNet-S2', 'Reference_Maps']:
        compress_directory(dirname)


if __name__ == '__main__':
    main()
