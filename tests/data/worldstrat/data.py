#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import hashlib
import os
import shutil
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

HR_DIR = 'hr_dataset'
LR_DIR = 'lr_dataset'

TRANSFORM = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)
LR_TRANSFORM = rasterio.Affine(
    9.553533791820828e-05,
    0.0,
    92.38122406971227,
    0.0,
    -9.096299266268611e-05,
    20.83381094772868,
)
CRS = rasterio.crs.CRS.from_epsg(4326)


def write_tiff(path: str, count: int, size: int, transform: rasterio.Affine) -> None:
    """Write a random uint16 GeoTIFF."""
    data = np.random.randint(0, 255, (count, size, size), dtype=np.uint16)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=count,
        dtype=np.uint16,
        transform=transform,
        crs=CRS,
    ) as dst:
        dst.write(data)


def create_dummy_worldstrat(root: str, img_size: int = 64) -> None:
    """Create dummy WorldStrat dataset."""
    os.makedirs(root, exist_ok=True)

    tiles = {
        'train': ['ASMSpotter-1-1-1', 'Landcover-773616'],
        'val': ['UNHCR-GHAs003590'],
        'test': ['Amnesty POI-1-1-1'],
    }

    metadata = []
    split_info = []

    # Generate 4 timesteps for the time series
    base_date = date(2021, 1, 1)
    dates = [base_date + timedelta(days=i * 30) for i in range(4)]

    # 1-based as in the real record
    write_order = [3, 1, 4, 2]

    for wrapper in (HR_DIR, LR_DIR):
        if os.path.exists(os.path.join(root, wrapper)):
            shutil.rmtree(os.path.join(root, wrapper))

    for split, tile_list in tiles.items():
        for tile in tile_list:
            hr_tile_dir = os.path.join(root, HR_DIR, tile)
            l1c_dir = os.path.join(root, LR_DIR, tile, 'L1C')
            l2a_dir = os.path.join(root, LR_DIR, tile, 'L2A')
            for directory in (hr_tile_dir, l1c_dir, l2a_dir):
                os.makedirs(directory, exist_ok=True)

            # High-res images (single timestep)
            write_tiff(
                os.path.join(hr_tile_dir, f'{tile}_ps.tiff'), 4, img_size, TRANSFORM
            )
            write_tiff(
                os.path.join(hr_tile_dir, f'{tile}_pan.tiff'), 1, img_size, TRANSFORM
            )

            # High-res RGBN (4 channels)
            hr_rgbn_png = np.random.randint(
                0, 255, (img_size, img_size, 4), dtype=np.uint8
            )
            rgbn_img = Image.fromarray(hr_rgbn_png, mode='RGBA')
            rgbn_img.save(os.path.join(hr_tile_dir, f'{tile}_rgb.png'))

            # Low-res RGBN
            write_tiff(
                os.path.join(hr_tile_dir, f'{tile}_rgbn.tiff'),
                4,
                img_size // 8,
                TRANSFORM,
            )

            # Time series data
            for n in write_order:
                write_tiff(
                    os.path.join(l1c_dir, f'{tile}-{n}-L1C_data.tiff'),
                    13,
                    img_size // 8,
                    LR_TRANSFORM,
                )
                write_tiff(
                    os.path.join(l2a_dir, f'{tile}-{n}-L2A_data.tiff'),
                    12,
                    img_size // 8,
                    LR_TRANSFORM,
                )

            # Metadata: one row per (tile, n)
            for n in write_order:
                metadata.append(
                    {
                        'tile': tile,
                        'n': n,
                        'lon': np.random.uniform(-180, 180),
                        'lat': np.random.uniform(-90, 90),
                        'lowres_date': dates[n - 1].strftime('%Y-%m-%d'),
                        'highres_date': dates[0].strftime('%Y-%m-%d'),
                    }
                )

            split_info.append({'tile': tile, 'split': split})

    pd.DataFrame(metadata).to_csv(os.path.join(root, 'metadata.csv'), index=False)
    pd.DataFrame(split_info).to_csv(
        os.path.join(root, 'stratified_train_val_test_split.csv'), index=False
    )


def create_archives(root: str) -> None:
    """Create zip archives and compute checksums."""
    # Each archive with wrapper directory
    archives = {
        'hr_dataset.zip': (HR_DIR, None),
        'lr_dataset_l1c.zip': (LR_DIR, 'L1C'),
        'lr_dataset_l2a.zip': (LR_DIR, 'L2A'),
    }

    checksums = {}

    for archive_name, (wrapper, subdir) in archives.items():
        archive_path = os.path.join(root, archive_name)
        wrapper_dir = os.path.join(root, wrapper)
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _, filenames in sorted(os.walk(wrapper_dir)):
                for filename in sorted(filenames):
                    src = os.path.join(dirpath, filename)
                    arcname = os.path.relpath(src, root)
                    if subdir is not None and os.path.basename(dirpath) != subdir:
                        continue
                    zf.write(src, arcname)

        checksums[archive_name] = compute_md5(archive_path)

    for csv_file in ['metadata.csv', 'stratified_train_val_test_split.csv']:
        checksums[csv_file] = compute_md5(os.path.join(root, csv_file))

    print('\nfile_info_dict entries:')
    for filename, checksum in checksums.items():
        name = filename.replace('.zip', '').replace('.csv', '')
        print(f"'{name}': {{")
        print(f"    'filename': '{filename}',")
        print(f"    'md5': '{checksum}',")
        print('},')


def compute_md5(filepath: str) -> str:
    """Compute MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


if __name__ == '__main__':
    root_dir = '.'
    create_dummy_worldstrat(root_dir)
    create_archives(root_dir)
