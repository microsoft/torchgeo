#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
from datetime import datetime, timedelta

import h5py
import numpy as np

# max rgb image
RGB_MAX = 255

NUM_SAMPLES = 3
NUM_CHANNELS = 3
SIZE = 64

np.random.seed(0)

data_dir = "dj417rh1007"
data_file = "2017_2019_images_pv_processed.hdf5"
splits = ["trainval", "test"]

# Create dataset file
data = np.random.randint(
    RGB_MAX, size=(NUM_SAMPLES, SIZE, SIZE, NUM_CHANNELS), dtype=np.int16
)
labels = np.random.random(size=(NUM_SAMPLES))

if __name__ == "__main__":
    # Remove old data
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    os.makedirs(data_dir)

    with h5py.File(os.path.join(data_dir, data_file), "w") as f:
        for split in splits:
            grp = f.create_group(split)
            grp.create_dataset("images_log", data=data)
            grp.create_dataset("pv_log", data=labels)

    # create time stamps
    for split in splits:
        time_stamps = np.array(
            [datetime.now() - timedelta(days=i) for i in range(NUM_SAMPLES)]
        )
        np.save(os.path.join(data_dir, f"times_{split}.npy"), time_stamps)

    # Compress data
    shutil.make_archive(data_dir, "zip", ".", data_dir)

    # Compute checksums
    with open(data_dir + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{data_dir}.zip: {md5}")
