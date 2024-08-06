#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import h5py
import numpy as np

SIZE = 32  # image width/height
NUM_CLASSES = 17
NUM_SAMPLES = 2

np.random.seed(0)

for split in ['training', 'validation', 'testing']:
    filename = f'{split}.h5'

    # Remove old data
    if os.path.exists(filename):
        os.remove(filename)

    # Random one hot encoding
    label = np.eye(NUM_CLASSES, dtype=np.uint8)[
        np.random.choice(NUM_CLASSES, NUM_SAMPLES)
    ]

    # Random images
    sen1 = np.random.randint(2, size=(NUM_SAMPLES, SIZE, SIZE, 8), dtype=np.uint8)
    sen2 = np.random.randint(2, size=(NUM_SAMPLES, SIZE, SIZE, 10), dtype=np.uint8)

    # Create datasets
    with h5py.File(filename, 'w') as f:
        f.create_dataset('label', data=label, compression='gzip', compression_opts=9)
        f.create_dataset('sen1', data=sen1, compression='gzip', compression_opts=9)
        f.create_dataset('sen2', data=sen2, compression='gzip', compression_opts=9)

    # Compute checksums
    with open(filename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(repr(split.replace('ing', '')) + ':', repr(md5) + ',')

for version in ['random', 'block', 'culture_10']:
    os.makedirs(version, exist_ok=True)
    shutil.copyfile('training.h5', os.path.join(version, 'training.h5'))
    shutil.copyfile('testing.h5', os.path.join(version, 'testing.h5'))
