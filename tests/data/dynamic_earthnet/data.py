#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import pandas as pd
import shutil
import tarfile

import numpy as np
import rasterio

directories = {'planet', 'sentinel1', 'sentinel2', 'metadata'}

samples = [
    {
        'planet_path': '/10N/26E-183N/1330_3107_13/',
        'label_path': '/labels/1330_3107_13_10N/Labels/Raster/10N-121W-39N-L3H-SR/10N-121W-39N-L3H-SR-2018_01_01.tif',
        'date': '2018-01',
    },
    {
        'planet_path': '/17N/9E-42N/2196_3885_13/',
        'label_path': '/labels/2196_3885_13_17N/Labels/Raster/17N-83W-9N-L3H-SR/17N-83W-9N-L3H-SR-2018-01-01.tif',
        'date': '2019-02',
    },
]

planet_dirs = ['PF-SR', 'PF-QA']

NUM_CLASSES = 7

SIZE = 32

# Dummy directory names and samples (already provided)
directories = {'planet', 'sentinel1', 'sentinel2', 'split_info', 'labels'}

splits = ['train', 'val', 'test']

planet_dirs = ['PF-SR', 'PF-QA']


def create_dummy_tiff(filepath: str, bands: int, label_mode=False):
    """Create a dummy raster with the specified number of bands.
    If label_mode=True, bands have 0 or 255 values only."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if label_mode:
        dtype = np.uint8
        data = np.random.choice([0, 255], size=(bands, SIZE, SIZE)).astype(dtype)
    else:
        dtype = np.int16
        data = np.random.randint(0, 255, size=(bands, SIZE, SIZE), dtype=dtype)

    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=bands,
        dtype=dtype,
        crs='+proj=latlong',
        compress='lzw',
    ) as dst:
        dst.write(data)


def get_days_in_month(date_str):
    """Get number of days in month from date string 'YYYY-MM'."""
    date = datetime.strptime(date_str, '%Y-%m')
    _, num_days = calendar.monthrange(date.year, date.month)
    return num_days


def create_split_files() -> pd.DataFrame:
    """Create train/val/test split files and parquet."""
    # Ensure metadata directory exists
    os.makedirs('split_info', exist_ok=True)

    # Define splits
    split_samples = {
        'train': [samples[0], samples[1]],  # both samples
        'val': [samples[0]],  # first sample
        'test': [samples[1]],  # second sample
    }

    # Create DataFrame for parquet
    df_data = []

    # Generate split files
    for split, split_samples in split_samples.items():
        lines = []
        for sample in split_samples:
            # Format: planet_path label_path date
            line = (
                f'{sample["planet_path"]}/PF-SR {sample["label_path"]} {sample["date"]}'
            )
            lines.append(line)

            # Add to DataFrame data
            df_data.append(
                {
                    'split': split,
                    'planet_path': f'{sample["planet_path"]}/PF-SR',
                    'label_path': sample['label_path'],
                    'year_month': sample['date'],
                }
            )

        # Write split file
        with open(os.path.join('split_info', f'{split}.txt'), 'w') as f:
            f.write('\n'.join(lines))

    # Create and save DataFrame
    df = pd.DataFrame(df_data)
    df['missing_label'] = False
    df['missing_s1'] = False
    df['missing_s2'] = False
    df['s1_path'] = df.apply(
        lambda row: f'sentinel1/{row["planet_path"].split("/")[3]}/{row["planet_path"].split("/")[3]}_{row["year_month"].replace("-", "_")}.tif',
        axis=1,
    )
    df['s2_path'] = df.apply(
        lambda row: f'sentinel2/{row["planet_path"].split("/")[3]}/{row["planet_path"].replace("-", "_")}.tif',
        axis=1,
    )
    df['planet_path'] = df['planet_path'].apply(lambda x: f'planet{x}')
    df['label_path'] = df['label_path'].apply(lambda x: x.lstrip('/'))
    df.to_parquet(os.path.join('split_info', 'splits.parquet'))

    return df


def main():
    # if directories exists remove them
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    # create the metadata
    df = create_split_files()

    # iterate over the metadata to create samples
    for i, row in df.iterrows():
        # # Create data for each modality
        # for sample in samples:
        num_days = pd.Period(row['year_month']).days_in_month
        planet_path = row['planet_path'].lstrip('/')
        # planet_base_path = os.path.dirname(planet_path)
        label_path = row['label_path'].lstrip('/')
        s1_path = row['s1_path']
        s2_path = row['s2_path']

        # Generate daily files for the whole month
        for day in range(1, num_days + 1):
            date = f'{row["year_month"]}-{day:02d}'

            # 1. Planet data (PF-SR → 4 bands, PF-QA → 1 band)
            for planet_dir in planet_dirs:
                bands = 4 if planet_dir == 'PF-SR' else 1
                tif_path = os.path.join(
                    os.path.dirname(planet_path), planet_dir, f'{date}.tif'
                )
                create_dummy_tiff(tif_path, bands=bands)

            # 2. Sentinel-1 data (8 band)
            tif_path = os.path.join(s1_path, f'{date}.tif')
            create_dummy_tiff(tif_path, bands=8)

            # 3. Sentinel-2 data (12 bands)
            tif_path = os.path.join(s2_path, f'{date}.tif')
            create_dummy_tiff(tif_path, bands=12)

            # 4. Labels (6 bands binary)
            tif_path = os.path.join(label_path, f'{date}.tif')
            create_dummy_tiff(tif_path, bands=NUM_CLASSES, label_mode=True)

    # 5) Create separate tarballs for each modality
    tar_info = [
        ('planet_pf_sr.tar.gz', 'planet', 'PF-SR'),
        ('planet_pf_qa.tar.gz', 'planet', 'PF-QA'),
        ('sentinel1.tar.gz', 'sentinel1', None),
        ('sentinel2.tar.gz', 'sentinel2', None),
        ('labels.tar.gz', 'labels', None),
        ('split_info.tar.gz', 'split_info', None),
    ]

    for tar_name, top_dir, sub_dir in tar_info:
        with tarfile.open(tar_name, 'w:gz') as tar:
            if sub_dir:
                # For planet data, include specific subdirectory but keep planet/ prefix
                for sample in samples:
                    add_dir = os.path.join(
                        top_dir, sample['planet_path'].lstrip('/'), sub_dir
                    )
                    if os.path.exists(add_dir):
                        # Include top_dir in arcname to maintain planet/ prefix
                        tar.add(add_dir)
            else:
                # For labels, include entire directory
                if os.path.exists(top_dir):
                    tar.add(top_dir)

        # compute md5sum of tarball
        with open(tar_name, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f'{tar_name}: {md5}')


if __name__ == '__main__':
    main()
