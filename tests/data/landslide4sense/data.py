#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import h5py
import numpy as np

NUM_CHANNELS = 14
SIZE = 32

np.random.seed(0)

BASE_DIR = Path(__file__).parent

files = [
    BASE_DIR / 'TrainData' / 'img' / 'image_1.h5',
    BASE_DIR / 'TrainData' / 'img' / 'image_2.h5',
    BASE_DIR / 'TrainData' / 'mask' / 'mask_1.h5',
    BASE_DIR / 'TrainData' / 'mask' / 'mask_2.h5',
    BASE_DIR / 'ValidData' / 'img' / 'image_1.h5',
    BASE_DIR / 'TestData' / 'img' / 'image_1.h5',
]

# Remove old data
for file in files:
    if os.path.exists(file):
        os.remove(file)

# Create train image files
for file in [
    BASE_DIR / 'TrainData' / 'img' / 'image_1.h5',
    BASE_DIR / 'TrainData' / 'img' / 'image_2.h5',
]:
    file.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.rand(SIZE, SIZE, NUM_CHANNELS).astype(np.float32)
    with h5py.File(file, 'w') as f:
        f.create_dataset('img', data=image)

# Create train mask files
for file in [
    BASE_DIR / 'TrainData' / 'mask' / 'mask_1.h5',
    BASE_DIR / 'TrainData' / 'mask' / 'mask_2.h5',
]:
    file.parent.mkdir(parents=True, exist_ok=True)
    mask = np.random.randint(0, 2, size=(SIZE, SIZE), dtype=np.uint8)
    with h5py.File(file, 'w') as f:
        f.create_dataset('mask', data=mask)

# Create validation image file
file = BASE_DIR / 'ValidData' / 'img' / 'image_1.h5'
file.parent.mkdir(parents=True, exist_ok=True)
image = np.random.rand(SIZE, SIZE, NUM_CHANNELS).astype(np.float32)
with h5py.File(file, 'w') as f:
    f.create_dataset('img', data=image)

# Create test image file
file = BASE_DIR / 'TestData' / 'img' / 'image_1.h5'
file.parent.mkdir(parents=True, exist_ok=True)
image = np.random.rand(SIZE, SIZE, NUM_CHANNELS).astype(np.float32)
with h5py.File(file, 'w') as f:
    f.create_dataset('img', data=image)
