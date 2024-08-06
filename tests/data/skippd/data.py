#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import zipfile
from datetime import datetime, timedelta

import h5py
import numpy as np

# max rgb image
RGB_MAX = 255

NUM_SAMPLES = 3
NUM_CHANNELS = 3
SIZE = 64
TIME_STEPS = 16

np.random.seed(0)

tasks = ['nowcast', 'forecast']
data_file = '2017_2019_images_pv_processed_{}.hdf5'
splits = ['trainval', 'test']


# Create dataset file

data = {
    'nowcast': np.random.randint(
        RGB_MAX, size=(NUM_SAMPLES, SIZE, SIZE, NUM_CHANNELS), dtype=np.int16
    ),
    'forecast': np.random.randint(
        RGB_MAX,
        size=(NUM_SAMPLES, TIME_STEPS, SIZE, SIZE, NUM_CHANNELS),
        dtype=np.int16,
    ),
}


labels = {
    'nowcast': np.random.random(size=(NUM_SAMPLES)),
    'forecast': np.random.random(size=(NUM_SAMPLES, TIME_STEPS)),
}


if __name__ == '__main__':
    for task in tasks:
        with h5py.File(data_file.format(task), 'w') as f:
            for split in splits:
                grp = f.create_group(split)
                grp.create_dataset('images_log', data=data[task])
                grp.create_dataset('pv_log', data=labels[task])

        # create time stamps
        for split in splits:
            time_stamps = np.array(
                [datetime.now() - timedelta(days=i) for i in range(NUM_SAMPLES)]
            )
            np.save(f'times_{split}_{task}.npy', time_stamps)

        # Compress data
        with zipfile.ZipFile(
            data_file.format(task).replace('.hdf5', '.zip'), 'w'
        ) as zip:
            for file in [
                data_file.format(task),
                f'times_trainval_{task}.npy',
                f'times_test_{task}.npy',
            ]:
                zip.write(file, arcname=file)

        # Compute checksums
        with open(data_file.format(task).replace('.hdf5', '.zip'), 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f'{task}: {md5}')
