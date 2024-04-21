#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os

import h5py
import numpy as np

NUM_CHANNELS = 2
SIZE = 32

np.random.seed(0)

filename = "earthquakes.h5"

splits = {
    "train": ["611645479", "611658170"],
    "validation": ["611684805", "611744956"],
    "test": ["611798698", "611818836"],
}

# Remove old data
if os.path.exists(filename):
    os.remove(filename)

# Create dataset file
data = np.random.randn(SIZE, SIZE, NUM_CHANNELS)
data = data.astype(np.float32)


with h5py.File(filename, "w") as f:
    for split, keys in splits.items():
        for key in keys:
            sample = f.create_group(key)
            sample.attrs.create(name="magnitude", data=np.float32(0.0))
            sample.attrs.create(name="split", data=split)
            for i in range(2):
                patch = sample.create_group(f"patch_{i}")
                patch.create_dataset("before", data=data)
                patch.create_dataset("pre", data=data)
                patch.create_dataset("post", data=data)

# Compute checksums
with open(filename, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(f"md5: {md5}")
