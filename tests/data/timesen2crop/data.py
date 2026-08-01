#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import zipfile

import numpy as np

# Sentinel-2 L2A surface reflectance is scaled by 10000
SENTINEL2_MAX = 10001

NUM_ACQUISITIONS = 5
NUM_BANDS = 9
NUM_CLASSES = 16
SAMPLES_PER_CLASS = 2

tiles = (
    '32TNT',
    '32TPT',
    '32TQT',
    '33TUM',
    '33TUN',
    '33TVM',
    '33TVN',
    '33TWM',
    '33TWN',
    '33TXN',
    '33UUP',
    '33UVP',
    '33UWP',
    '33UWQ',
    '33UXP',
    '2019_33UVP',
)

header = 'B1,B2,B3,B4,B5,B6,B7,B8,B9,Flag'
dates = ('20171001', '20171221', '20180304', '20180528', '20180821')
dates_2019 = ('20181008', '20190131', '20190416', '20190625', '20190824')

np.random.seed(0)

directory = 'TimeSen2Crop'

# Remove old data
if os.path.exists(directory):
    shutil.rmtree(directory)

# Create dataset files
for tile in tiles:
    os.makedirs(os.path.join(directory, tile))
    with open(os.path.join(directory, tile, 'dates.csv'), 'w', newline='\n') as f:
        f.write('acquisition_date\n')
        f.writelines(
            f'{date}\n' for date in (dates_2019 if tile == '2019_33UVP' else dates)
        )

    for label in range(NUM_CLASSES):
        os.makedirs(os.path.join(directory, tile, str(label)))
        for sample in range(SAMPLES_PER_CLASS):
            data = np.random.randint(
                SENTINEL2_MAX, size=(NUM_ACQUISITIONS, NUM_BANDS), dtype=np.int64
            )
            flag = np.random.randint(4, size=(NUM_ACQUISITIONS, 1), dtype=np.int64)
            array = np.hstack([data, flag])
            path = os.path.join(directory, tile, str(label), f'{sample}.csv')
            with open(path, 'w', newline='\n') as f:
                f.write(f'{header}\n')
                f.writelines(','.join(map(str, row)) + '\n' for row in array)

# Create zip file
filename = f'{directory}.zip'
if os.path.exists(filename):
    os.remove(filename)

with zipfile.ZipFile(filename, 'w') as f:
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            f.write(path, arcname=os.path.relpath(path))

# Compute checksum
with open(filename, 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(f'{filename}: {md5}')
