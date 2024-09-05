#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import random

import h5py
import numpy as np

# Sentinel-2 is 12-bit with range 0-4095
SENTINEL2_MAX = 4096

NUM_CHANNELS = 12
NUM_CLASSES = 2
SIZE = 32

np.random.seed(0)
random.seed(0)

filenames = ['512x512.hdf5', 'chabud_test.h5']
fold_mapping = {'train': [1, 2, 3, 4], 'val': [0], 'test': ['chabud']}

uris = [
    'feb08801-64b1-4d11-a3fc-0efaad1f4274_0',
    'e4d4dbcb-dd92-40cf-a7fe-fda8dd35f367_1',
    '9fc8c1f4-1858-47c3-953e-1dc8b179a',
    '3a1358a2-6155-445a-a269-13bebd9741a8_0',
    '2f8e659c-f457-4527-a57f-bffc3bbe0baa_0',
    '299ee670-19b1-4a76-bef3-34fd55580711_1',
    '05cfef86-3e27-42be-a0cb-a61fe2f89e40_0',
    '0328d12a-4ad8-4504-8ac5-70089db10b4e_1',
    '04800581-b540-4f9b-9df8-7ee433e83f46_0',
    '108ae2a9-d7d6-42f7-b89a-90bb75c23ccb_0',
    '29413474-04b8-4bb1-8b89-fd640023d4a6_0',
    '43f2e60a-73b4-4f33-b99e-319d892fcab4_0',
]
folds = random.choices(fold_mapping['train'], k=4) + [0] * 4 + ['chabud'] * 4
files = ['512x512.hdf5'] * 8 + ['chabud_test.h5'] * 4

# Remove old data
for filename in filenames:
    if os.path.exists(filename):
        os.remove(filename)

# Create dataset file
data = np.random.randint(
    SENTINEL2_MAX, size=(SIZE, SIZE, NUM_CHANNELS), dtype=np.uint16
)
gt = np.random.randint(NUM_CLASSES, size=(SIZE, SIZE, 1), dtype=np.uint16)

for filename, uri, fold in zip(files, uris, folds):
    with h5py.File(filename, 'a') as f:
        sample = f.create_group(uri)
        sample.attrs.create(
            name='fold', data=np.int64(fold) if fold != 'chabud' else fold
        )
        sample.create_dataset
        sample.create_dataset('pre_fire', data=data)
        sample.create_dataset('post_fire', data=data)
        sample.create_dataset('mask', data=gt)

# Compute checksums
for filename in filenames:
    with open(filename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{filename} md5: {md5}')
