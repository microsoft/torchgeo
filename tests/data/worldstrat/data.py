#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import tarfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


def create_dummy_worldstrat(root: str, img_size: int = 64) -> None:
    """Create dummy WorldStrat dataset."""
    os.makedirs(root, exist_ok=True)

    tiles = {'train': ['AOI001', 'AOI002'], 'val': ['AOI003'], 'test': ['AOI004']}

    metadata = []
    split_info = []

    # Generate 4 dates for time series
    base_date = datetime(2021, 1, 1)
    dates = [base_date + timedelta(days=i * 30) for i in range(4)]

    for split, tile_list in tiles.items():
        for tile in tile_list:
            if os.path.exists(os.path.join(root, tile)):
                shutil.rmtree(os.path.join(root, tile))
            tile_dir = os.path.join(root, tile)
            l1c_dir = os.path.join(tile_dir, 'L1C')
            l2a_dir = os.path.join(tile_dir, 'L2A')
            os.makedirs(l1c_dir, exist_ok=True)
            os.makedirs(l2a_dir, exist_ok=True)

            # High-res images (single timestep)
            hr_ps = np.random.randint(0, 255, (4, img_size, img_size), dtype=np.uint16)
            with rasterio.open(
                os.path.join(tile_dir, f'{tile}_ps.tiff'),
                'w',
                driver='GTiff',
                height=img_size,
                width=img_size,
                count=4,
                dtype=np.uint16,
            ) as dst:
                dst.write(hr_ps)

            hr_pan = np.random.randint(0, 255, (1, img_size, img_size), dtype=np.uint16)
            with rasterio.open(
                os.path.join(tile_dir, f'{tile}_pan.tiff'),
                'w',
                driver='GTiff',
                height=img_size,
                width=img_size,
                count=1,
                dtype=np.uint16,
            ) as dst:
                dst.write(hr_pan)

            # High-res RGBN (4 channels)
            hr_rgbn_png = np.random.randint(
                0, 255, (img_size, img_size, 4), dtype=np.uint8
            )
            rgbn_img = Image.fromarray(hr_rgbn_png, mode='RGBA')
            rgbn_img.save(os.path.join(tile, f'{tile}_rgb.png'))

            # Low-res RGBN
            lr_rgbn = np.random.randint(
                0, 255, (4, img_size // 4, img_size // 4), dtype=np.uint16
            )
            with rasterio.open(
                os.path.join(tile_dir, f'{tile}_rgbn.tiff'),
                'w',
                driver='GTiff',
                height=img_size // 8,
                width=img_size // 8,
                count=4,
                dtype=np.uint16,
            ) as dst:
                dst.write(lr_rgbn)

            # Time series data
            for date in dates:
                date_str = date.strftime('%Y%m%d')

                # L1C (13 bands)
                l1c = np.random.randint(
                    0, 255, (13, img_size // 8, img_size // 8), dtype=np.uint16
                )
                with rasterio.open(
                    os.path.join(l1c_dir, f'{tile}_{date_str}_L1C_data.tiff'),
                    'w',
                    driver='GTiff',
                    height=img_size // 8,
                    width=img_size // 8,
                    count=13,
                    dtype=np.uint16,
                ) as dst:
                    dst.write(l1c)

                # L2A (12 bands)
                l2a = np.random.randint(
                    0, 255, (12, img_size // 8, img_size // 8), dtype=np.uint16
                )
                with rasterio.open(
                    os.path.join(l2a_dir, f'{tile}_{date_str}_L2A_data.tiff'),
                    'w',
                    driver='GTiff',
                    height=img_size // 8,
                    width=img_size // 8,
                    count=12,
                    dtype=np.uint16,
                ) as dst:
                    dst.write(l2a)

            # Metadata with date
            for date in dates:
                metadata.append(
                    {
                        'tile': tile,
                        'lon': np.random.uniform(-180, 180),
                        'lat': np.random.uniform(-90, 90),
                        'lowres_date': date.strftime('%Y-%m-%d'),
                        'highres_date': date.strftime('%Y-%m-%d'),
                    }
                )

            split_info.append({'tile': tile, 'split': split})

    pd.DataFrame(metadata).to_csv(os.path.join(root, 'metadata.csv'), index=True)
    pd.DataFrame(split_info).to_csv(
        os.path.join(root, 'stratified_train_val_test_split.csv'), index=False
    )


def create_archives(root: str) -> None:
    """Create compressed archives and compute checksums."""
    # Create archive structure
    archives = {
        'hr_dataset.tar.gz': ['_ps.tiff', '_pan.tiff', '_rgbn.tiff', '_rgb.png'],
        'lr_dataset_l1c.tar.gz': ['L1C'],
        'lr_dataset_l2a.tar.gz': ['L2A'],
    }

    checksums = {}

    # Create each archive
    for archive_name, patterns in archives.items():
        archive_path = os.path.join(root, archive_name)
        with tarfile.open(archive_path, 'w:gz') as tar:
            for aoi in os.listdir(root):
                aoi_path = os.path.join(root, aoi)
                if not os.path.isdir(aoi_path) or aoi.startswith('.'):
                    continue

                # Add files matching patterns
                for pattern in patterns:
                    if pattern.startswith('_'):  # High-res files
                        src = os.path.join(aoi_path, f'{aoi}{pattern}')
                        if os.path.exists(src):
                            tar.add(src, os.path.join(aoi, os.path.basename(src)))
                    else:  # L1C/L2A directories
                        src_dir = os.path.join(aoi_path, pattern)
                        if os.path.exists(src_dir):
                            for f in os.listdir(src_dir):
                                src = os.path.join(src_dir, f)
                                tar.add(src, os.path.join(aoi, pattern, f))

        checksums[archive_name] = compute_md5(archive_path)

    # Add CSV files
    for csv_file in ['metadata.csv', 'stratified_train_val_test_split.csv']:
        checksums[csv_file] = compute_md5(os.path.join(root, csv_file))

    # Print checksums in format matching file_info_dict
    print('\nfile_info_dict entries:')
    for filename, checksum in checksums.items():
        name = filename.replace('.tar.gz', '').replace('.csv', '')
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
