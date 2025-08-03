#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import h5py
import numpy as np

SIZE = 32  # image width/height
NUM_CLASSES = 17
NUM_SAMPLES = 1

np.random.seed(0)

for split in ['train', 'val', 'test']:
    filename = f'lcz_{split}.h5'

    # Random one hot encoding
    label = np.eye(NUM_CLASSES, dtype='|u1')[np.random.choice(NUM_CLASSES, NUM_SAMPLES)]

    # Random images
    sen1 = np.random.random(size=(NUM_SAMPLES, SIZE, SIZE, 8)).astype('<f4')
    sen2 = np.random.random(size=(NUM_SAMPLES, SIZE, SIZE, 10)).astype('<f4')

    # Create datasets
    with h5py.File(filename, 'w') as f:
        f.create_dataset('label', data=label, compression='gzip', compression_opts=9)
        f.create_dataset('sen1', data=sen1, compression='gzip', compression_opts=9)
        f.create_dataset('sen2', data=sen2, compression='gzip', compression_opts=9)
