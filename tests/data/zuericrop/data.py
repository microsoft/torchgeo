#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import h5py

import numpy as np


# Sentinel-2 is 12-bit with range 0-4095
SENTINEL2_MAX = 4095

NUM_SAMPLES = 2
NUM_CHANNELS = 9
SIZE = 24
NUM_CLASSES = 10

np.random.seed(0)

data_file = "ZueriCrop.hdf5"
labels_file = "labels.csv"

# Remove old data
os.remove(data_file)
os.remove(labels_file)

# Create empty labels file
with open(labels_file, "w") as f:
    pass

# Create dataset file
data = np.random.randint(
    SENTINEL2_MAX,
    size=(NUM_SAMPLES, 1, SIZE, SIZE, NUM_CHANNELS),
    dtype=np.int16
)
data = data.astype(np.float64)
gt = np.random.randint(
    NUM_CLASSES,
    size=(NUM_SAMPLES, SIZE, SIZE, 1),
    dtype=np.int16
)
gt_instance = np.random.randint(
    NUM_CLASSES,
    size=(NUM_SAMPLES, SIZE, SIZE, 1),
    dtype=np.int32
)

with h5py.File(data_file, "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("gt", data=gt)
    f.create_dataset("gt_instance", data=gt_instance)
