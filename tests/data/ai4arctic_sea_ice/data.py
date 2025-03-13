#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import numpy as np
import xarray as xr
import pandas as pd
import tarfile
import hashlib
import shutil
from datetime import datetime, timedelta


def create_dummy_nc_file(filepath: str, is_reference: bool = False):
    """Create dummy netCDF file matching original dataset structure."""

    # Define dimensions
    dims = {
        'sar_lines': 12,
        'sar_samples': 9,
        'sar_sample_2dgrid_points': 3,
        'sar_line_2dgrid_points': 4,
        '2km_grid_lines': 5,
        '2km_grid_samples': 6,
    }

    # Create variables with realistic dummy data
    data_vars = {
        # SAR variables (full resolution)
        'nersc_sar_primary': (
            ('sar_lines', 'sar_samples'),
            np.random.normal(-20, 5, (dims['sar_lines'], dims['sar_samples'])).astype(
                np.float32
            ),
        ),
        'nersc_sar_secondary': (
            ('sar_lines', 'sar_samples'),
            np.random.normal(-25, 5, (dims['sar_lines'], dims['sar_samples'])).astype(
                np.float32
            ),
        ),
        # Grid coordinates
        'sar_grid2d_latitude': (
            ('sar_sample_2dgrid_points', 'sar_line_2dgrid_points'),
            np.random.uniform(
                60,
                80,
                (dims['sar_sample_2dgrid_points'], dims['sar_line_2dgrid_points']),
            ).astype(np.float64),
        ),
        'sar_grid2d_longitude': (
            ('sar_sample_2dgrid_points', 'sar_line_2dgrid_points'),
            np.random.uniform(
                -60,
                0,
                (dims['sar_sample_2dgrid_points'], dims['sar_line_2dgrid_points']),
            ).astype(np.float64),
        ),
        # Weather variables (2km grid)
        'u10m_rotated': (
            ('2km_grid_lines', '2km_grid_samples'),
            np.random.normal(
                0, 5, (dims['2km_grid_lines'], dims['2km_grid_samples'])
            ).astype(np.float32),
        ),
        'v10m_rotated': (
            ('2km_grid_lines', '2km_grid_samples'),
            np.random.normal(
                0, 5, (dims['2km_grid_lines'], dims['2km_grid_samples'])
            ).astype(np.float32),
        ),
        # AMSR2 variables (6.9, 7.3, 10.7, 23.8, 36.5, 89.0 GHz, h, v)
        **{
            f'btemp_{freq}{pol}': (
                ('2km_grid_lines', '2km_grid_samples'),
                np.random.normal(
                    250, 20, (dims['2km_grid_lines'], dims['2km_grid_samples'])
                ).astype(np.float32),
            )
            for freq in ['6_9', '7_3']
            for pol in ['h', 'v']
        },
        # Add distance map
        'distance_map': (
            ('sar_lines', 'sar_samples'),
            np.random.uniform(0, 10, (dims['sar_lines'], dims['sar_samples'])).astype(
                np.float32
            ),
            {
                'long_name': 'Distance to land zones numbered with ids ranging from 0 to N',
                'zonal_range_description': '\ndist_id; dist_range_km\n0; land\n1; 0 -> 0.5\n2; 0.5 -> 1\n3; 1 -> 2\n4; 2 -> 4\n5; 4 -> 8\n6; 8 -> 16\n7; 16 -> 32\n8; 32 -> 64\n9; 64 -> 128\n10; >128',
            },
        ),
    }

    # Add target variables if reference file
    if is_reference:
        data_vars.update(
            {
                'SOD': (
                    ('sar_lines', 'sar_samples'),
                    np.random.randint(
                        0, 6, (dims['sar_lines'], dims['sar_samples'])
                    ).astype(np.uint8),
                ),
                'SIC': (
                    ('sar_lines', 'sar_samples'),
                    np.random.randint(
                        0, 11, (dims['sar_lines'], dims['sar_samples'])
                    ).astype(np.uint8),
                ),
                'FLOE': (
                    ('sar_lines', 'sar_samples'),
                    np.random.randint(
                        0, 7, (dims['sar_lines'], dims['sar_samples'])
                    ).astype(np.uint8),
                ),
            }
        )

    # Create dataset with correct attributes
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={
            'scene_id': os.path.basename(filepath),
            'original_id': f'S1A_EW_GRDM_1SDH_{os.path.basename(filepath)}',
            'ice_service': 'dmi' if 'dmi' in filepath else 'cis',
            'flip': 0,
            'pixel_spacing': 80,
        },
    )

    # Save to netCDF file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ds.to_netcdf(filepath)


def create_metadata_csv(root_dir: str, n_train: int = 3, n_test: int = 2):
    """Create metadata CSV file."""
    records = []

    # Generate dates
    base_date = datetime(2021, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_train + n_test)]

    # Create train records
    for i in range(n_train):
        date_str = dates[i].strftime('%Y%m%dT%H%M%S')
        service = 'dmi' if i % 2 == 0 else 'cis'
        path = f'train/{date_str}_{service}_prep.nc'
        records.append(
            {
                'input_path': path,
                'reference_path': None,
                'date': dates[i],
                'ice_service': service,
                'split': 'train',
                'region_id': 'SGRDIFOXE' if service == 'cis' else 'North_RIC',
            }
        )

    # Create test records
    for i in range(n_test):
        date_str = dates[n_train + i].strftime('%Y%m%dT%H%M%S')
        service = 'dmi' if i % 2 == 0 else 'cis'
        input_path = f'test/{date_str}_{service}_prep.nc'
        ref_path = f'test/{date_str}_{service}_prep_reference.nc'
        records.append(
            {
                'input_path': input_path,
                'reference_path': ref_path,
                'date': dates[n_train + i],
                'ice_service': service,
                'split': 'test',
                'region_id': 'SGRDIFOXE' if service == 'cis' else 'North_RIC',
            }
        )

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(root_dir, 'metadata.csv'), index=False)
    return df


def main():
    """Create complete dummy dataset."""
    root_dir = '.'
    n_train = 3
    n_test = 2

    # Create metadata
    df = create_metadata_csv(root_dir, n_train, n_test)

    # Create train files
    train_files = df[df['split'] == 'train']['input_path']
    for f in train_files:
        create_dummy_nc_file(os.path.join(root_dir, f), is_reference=True)

    # Create test files
    test_files = df[df['split'] == 'test']
    for _, row in test_files.iterrows():
        create_dummy_nc_file(
            os.path.join(root_dir, row['input_path']), is_reference=False
        )
        create_dummy_nc_file(
            os.path.join(root_dir, row['reference_path']), is_reference=True
        )

    # Create and split train tarball
    shutil.make_archive('train', 'gztar', '.', 'train')

    with open('train.tar.gz', 'rb') as f:
        content = f.read()

    # Split into two chunks
    chunk1 = content[: len(content) // 2]
    chunk2 = content[len(content) // 2 :]

    with open('train.tar.gzaa', 'wb') as g:
        g.write(chunk1)
    with open('train.tar.gzab', 'wb') as g:
        g.write(chunk2)

    # Remove original tarball
    os.remove('train.tar.gz')

    with tarfile.open('test.tar.gz', 'w:gz') as tar:
        tar.add('test')

    # compute md5sum
    def md5(fname: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fname, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    print(f'MD5 checksum train.gzaa: {md5("train.tar.gzaa")}')
    print(f'MD5 checksum train.gzab: {md5("train.tar.gzab")}')
    print(f'MD5 checksum test.gz: {md5("test.tar.gz")}')
    print(f'MD5 checksum metadata: {md5("metadata.csv")}')


if __name__ == '__main__':
    main()
