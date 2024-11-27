#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for running the benchmark script over a sweep of different options."""

import itertools
import os
import subprocess
import time

EPOCH_SIZE = 4096

SEED_OPTIONS = [0, 1, 2]
CACHE_OPTIONS = [True, False]
BATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256, 512]

# path to a directory containing Landsat 8 GeoTIFFs
LANDSAT_DATA_ROOT = ''

# path to a directory containing CDL GeoTIFF(s)
CDL_DATA_ROOT = ''

total_num_experiments = len(SEED_OPTIONS) * len(CACHE_OPTIONS) * len(BATCH_SIZE_OPTIONS)

if __name__ == '__main__':
    # With 6 workers, this will use ~60% of available RAM
    os.environ['GDAL_CACHEMAX'] = '10%'

    tic = time.time()
    for i, (cache, batch_size, seed) in enumerate(
        itertools.product(CACHE_OPTIONS, BATCH_SIZE_OPTIONS, SEED_OPTIONS)
    ):
        print(f'\n{i}/{total_num_experiments} -- {time.time() - tic}')
        tic = time.time()
        command: list[str] = [
            'python',
            'benchmark.py',
            '--landsat-root',
            LANDSAT_DATA_ROOT,
            '--cdl-root',
            CDL_DATA_ROOT,
            '--num-workers',
            '6',
            '--batch-size',
            str(batch_size),
            '--epoch-size',
            str(EPOCH_SIZE),
            '--seed',
            str(seed),
            '--verbose',
        ]

        if cache:
            command.append('--cache')

        subprocess.call(command)
