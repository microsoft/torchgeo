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

ROOT = '.'
DATA_DIR = 'dfc25_track2_trainval'

TRAIN_FILE = 'train_setlevel.txt'
HOLDOUT_FILE = 'holdout_setlevel.txt'
VAL_FILE = 'val_setlevel.txt'

TRAIN_IDS = [
    'bata-explosion_00000049',
    'bata-explosion_00000014',
    'bata-explosion_00000047',
]
HOLDOUT_IDS = ['turkey-earthquake_00000413']
VAL_IDS = ['val-disaster_00000001', 'val-disaster_00000002']

SIZE = 32
crs = CRS.from_wkt("""
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AXIS["Latitude",NORTH],
    AXIS["Longitude",EAST],
    AUTHORITY["EPSG","4326"]]
""")
transform = Affine(
    4.572424737366368e-06,
    0.0,
    9.796201318793191,
    0.0,
    -4.572424813937713e-06,
    1.846511153,
)


def make_dirs() -> None:
    paths = [
        os.path.join(ROOT, DATA_DIR),
        os.path.join(ROOT, DATA_DIR, 'train', 'pre-event'),
        os.path.join(ROOT, DATA_DIR, 'train', 'post-event'),
        os.path.join(ROOT, DATA_DIR, 'train', 'target'),
        os.path.join(ROOT, DATA_DIR, 'val', 'pre-event'),
        os.path.join(ROOT, DATA_DIR, 'val', 'post-event'),
        os.path.join(ROOT, DATA_DIR, 'val', 'target'),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_list_file(filename: str, ids: list[str]) -> None:
    file_path = os.path.join(ROOT, DATA_DIR, filename)
    with open(file_path, 'w') as f:
        for sid in ids:
            f.write(f'{sid}\n')


def write_tif(filepath: str, channels: int, classes: int) -> None:
    data = np.random.randint(0, classes, (channels, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=channels,
        crs=crs,
        dtype=data.dtype,
        compress='lzw',
        transform=transform,
    ) as dst:
        dst.write(data)


def populate_data(ids: list[str], dir_name: str, with_target: bool = True) -> None:
    for sid in ids:
        pre_path = os.path.join(
            ROOT, DATA_DIR, dir_name, 'pre-event', f'{sid}_pre_disaster.tif'
        )
        write_tif(pre_path, channels=3, classes=256)
        post_path = os.path.join(
            ROOT, DATA_DIR, dir_name, 'post-event', f'{sid}_post_disaster.tif'
        )
        write_tif(post_path, channels=1, classes=256)
        if with_target:
            target_path = os.path.join(
                ROOT, DATA_DIR, dir_name, 'target', f'{sid}_building_damage.tif'
            )
            write_tif(target_path, channels=1, classes=2)


def main() -> None:
    make_dirs()

    # Write the ID lists to text files
    write_list_file(TRAIN_FILE, TRAIN_IDS)
    write_list_file(HOLDOUT_FILE, HOLDOUT_IDS)
    write_list_file(VAL_FILE, VAL_IDS)

    # Generate TIF files for the train (with target) and val (no target) splits
    populate_data(TRAIN_IDS, 'train', with_target=True)
    populate_data(HOLDOUT_IDS, 'train', with_target=True)
    populate_data(VAL_IDS, 'val', with_target=False)

    # zip and compute md5
    zip_filename = os.path.join(ROOT, 'dfc25_track2_trainval')
    shutil.make_archive(zip_filename, 'zip', ROOT, DATA_DIR)

    def md5(fname: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fname, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    md5sum = md5(zip_filename + '.zip')
    print(f'MD5 checksum: {md5sum}')


if __name__ == '__main__':
    main()
