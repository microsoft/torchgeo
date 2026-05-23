#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
from PIL import Image

BATCH = 2
SIZE = 64  # image width/height

np.random.seed(0)

directories = [
    'Onera Satellite Change Detection dataset - Images',
    'Onera Satellite Change Detection dataset - Train Labels',
    'Onera Satellite Change Detection dataset - Val Labels',
    'Onera Satellite Change Detection dataset - Test Labels',
]
bands = [
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B09',
    'B10',
    'B11',
    'B12',
    'B8A',
]

# Remove old data
for directory in directories:
    filename = f'{directory}.zip'

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(directory):
        shutil.rmtree(directory)

split_to_dir = {'train': 1, 'val': 2, 'test': 3}
splits = [('train', BATCH * 2), ('val', BATCH), ('test', BATCH)]
for split, count in splits:
    for j in range(count):
        subdir = f'{split}{j}'

        size = (SIZE + np.random.randint(3), SIZE + np.random.randint(3))

        # Create images
        for rect in ['imgs_1_rect', 'imgs_2_rect']:
            directory = os.path.join(directories[0], subdir, rect)
            os.makedirs(directory)

            for band in bands:
                filename = os.path.join(directory, f'{band}.tif')
                arr = np.random.randint(
                    np.iinfo(np.uint16).max, size=size, dtype=np.uint16
                )
                img = Image.fromarray(arr)
                img.save(filename)

        filename = os.path.join(directories[0], subdir, 'dates.txt')
        with open(filename, 'w') as f:
            for key, value in [('date_1', '20161130'), ('date_2', '20170829')]:
                f.write(f'{key}: {value}\n')

        # Create labels
        directory = os.path.join(directories[split_to_dir[split]], subdir, 'cm')
        os.makedirs(directory)
        filename = os.path.join(directory, 'cm.png')
        arr = np.random.randint(np.iinfo(np.uint8).max, size=size, dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(filename)

for directory in directories:
    # Compress data
    shutil.make_archive(directory, 'zip', '.', directory)
