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

filename = 'train_eval.hdf5'
fold_mapping = {'train': [1, 2, 3, 4], 'val': [0]}

uris = [
    'feb08801-64b1-4d11-a3fc-0efaad1f4274_0',
    'e4d4dbcb-dd92-40cf-a7fe-fda8dd35f367_1',
    '9fc8c1f4-1858-47c3-953e-1dc8b179a',
    '3a1358a2-6155-445a-a269-13bebd9741a8_0',
    '2f8e659c-f457-4527-a57f-bffc3bbe0baa_0',
    '299ee670-19b1-4a76-bef3-34fd55580711_1',
    '05cfef86-3e27-42be-a0cb-a61fe2f89e40_0',
    '0328d12a-4ad8-4504-8ac5-70089db10b4e_1',
]
folds = [
    random.sample(fold_mapping['train'], 1)[0],
    random.sample(fold_mapping['train'], 1)[0],
    random.sample(fold_mapping['train'], 1)[0],
    random.sample(fold_mapping['train'], 1)[0],
    random.sample(fold_mapping['val'], 1)[0],
    random.sample(fold_mapping['val'], 1)[0],
    random.sample(fold_mapping['val'], 1)[0],
    random.sample(fold_mapping['val'], 1)[0],
]

# Remove old data
if os.path.exists(filename):
    os.remove(filename)

# Create dataset file
data = np.random.randint(
    SENTINEL2_MAX, size=(SIZE, SIZE, NUM_CHANNELS), dtype=np.uint16
)
data = data.astype(np.uint16)
gt = np.random.randint(NUM_CLASSES, size=(SIZE, SIZE, 1), dtype=np.uint16)

with h5py.File(filename, 'w') as f:
    for uri, fold in zip(uris, folds):
        sample = f.create_group(uri)
        sample.attrs.create(name='fold', data=np.int64(fold))
        sample.create_dataset
        sample.create_dataset('pre_fire', data=data)
        sample.create_dataset('post_fire', data=data)
        sample.create_dataset('mask', data=gt)

# Compute checksums
with open(filename, 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(f'md5: {md5}')
