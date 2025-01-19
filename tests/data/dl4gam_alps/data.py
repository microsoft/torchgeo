#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# define the patch size
PATCH_SIZE = 16

# create a random generator
rg = np.random.RandomState(42)


def create_dummy_sample(fp: str | Path) -> None:
    # create the random S2 bands data; make the last two bands as binary masks
    band_data = rg.randint(
        low=0, high=10000, dtype=np.int16, size=(15, PATCH_SIZE, PATCH_SIZE)
    )
    band_data[-2:] = (band_data[-2:] > 5000).astype(np.int16)

    data_dict = {
        'band_data': {
            'dims': ('band', 'y', 'x'),
            'data': band_data,
            'attrs': {
                'long_name': [
                    'B1',
                    'B2',
                    'B3',
                    'B4',
                    'B5',
                    'B6',
                    'B7',
                    'B8',
                    'B8A',
                    'B9',
                    'B10',
                    'B11',
                    'B12',
                    'CLOUDLESS_MASK',
                    'FILL_MASK',
                ],
                '_FillValue': -9999,
            },
        },
        'mask_all_g_id': {  # glaciers mask (with -1 for no-glacier and GLACIER_ID for glacier)
            'dims': ('y', 'x'),
            'data': rg.choice([-1, 8, 9, 30, 35], size=(PATCH_SIZE, PATCH_SIZE)).astype(
                np.int32
            ),
            'attrs': {'_FillValue': -1},
        },
        'mask_debris': {
            'dims': ('y', 'x'),
            'data': (rg.random((PATCH_SIZE, PATCH_SIZE)) > 0.5).astype(np.int8),
            'attrs': {'_FillValue': -1},
        },
    }

    # add the additional variables
    for v in [
        'dem',
        'slope',
        'aspect',
        'planform_curvature',
        'profile_curvature',
        'terrain_ruggedness_index',
        'dhdt',
        'v',
    ]:
        data_dict[v] = {
            'dims': ('y', 'x'),
            'data': (rg.random((PATCH_SIZE, PATCH_SIZE)) * 100).astype(np.float32),
            'attrs': {'_FillValue': -9999},
        }

    # create the xarray dataset and save it
    nc = xr.Dataset.from_dict(data_dict)
    nc.to_netcdf(fp)


def create_splits_df(fp: str | Path) -> pd.DataFrame:
    # create a dataframe with the splits for the 4 glaciers
    splits_df = pd.DataFrame(
        {
            'entry_id': ['g_0008', 'g_0009', 'g_0030', 'g_0035'],
            'split_1': ['fold_train', 'fold_train', 'fold_valid', 'fold_test'],
            'split_2': ['fold_train', 'fold_valid', 'fold_train', 'fold_test'],
            'split_3': ['fold_train', 'fold_valid', 'fold_test', 'fold_train'],
            'split_4': ['fold_test', 'fold_valid', 'fold_train', 'fold_train'],
            'split_5': ['fold_test', 'fold_train', 'fold_train', 'fold_valid'],
        }
    )

    splits_df.to_csv(fp_splits, index=False)
    print(f'Splits dataframe saved to {fp_splits}')
    return splits_df


if __name__ == '__main__':
    # prepare the paths
    fp_splits = Path('splits.csv')
    fp_dir_ds_small = Path('dataset_small')
    fp_dir_ds_large = Path('dataset_large')

    # cleanup
    fp_splits.unlink(missing_ok=True)
    fp_dir_ds_small.with_suffix('.tar.gz').unlink(missing_ok=True)
    fp_dir_ds_large.with_suffix('.tar.gz').unlink(missing_ok=True)
    shutil.rmtree(fp_dir_ds_small, ignore_errors=True)
    shutil.rmtree(fp_dir_ds_large, ignore_errors=True)

    # create the splits dataframe
    split_df = create_splits_df(fp_splits)

    # create the two datasets versions (small and large) with 1 and 2 patches per glacier, respectively
    for fp_dir, num_patches in zip([fp_dir_ds_small, fp_dir_ds_large], [1, 2]):
        for glacier_id in split_df.entry_id:
            for i in range(num_patches):
                fp = fp_dir / glacier_id / f'{glacier_id}_patch_{i}.nc'
                fp.parent.mkdir(parents=True, exist_ok=True)
                create_dummy_sample(fp=fp)

    # archive the datasets
    for fp_dir in [fp_dir_ds_small, fp_dir_ds_large]:
        shutil.make_archive(str(fp_dir), 'gztar', fp_dir)

    # compute checksums
    for fp in [
        fp_dir_ds_small.with_suffix('.tar.gz'),
        fp_dir_ds_large.with_suffix('.tar.gz'),
        fp_splits,
    ]:
        with open(fp, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f'md5 for {fp}: {md5}')
