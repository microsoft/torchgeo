#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image

DTYPE = np.float32
SIZE = 2

np.random.seed(0)

all_bands = (
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B8A',
    'B09',
    'B11',
    'B12',
    'CLD',
)

for tile in range(1):
    directory = os.path.join('data', str(tile))
    os.makedirs(directory, exist_ok=True)

    arr = np.random.randint(np.iinfo(np.int32).max, size=(SIZE, SIZE), dtype=np.int32)
    img = Image.fromarray(arr)
    img.save(os.path.join(directory, f'{tile}_field_id.tif'))

    arr = np.random.randint(np.iinfo(np.uint8).max, size=(SIZE, SIZE), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(os.path.join(directory, f'{tile}_label.tif'))

    for date in ['20190606']:
        directory = os.path.join(directory, date)
        os.makedirs(directory, exist_ok=True)

        for band in all_bands:
            arr = np.random.rand(SIZE, SIZE).astype(DTYPE) * np.finfo(DTYPE).max
            img = Image.fromarray(arr)
            img.save(os.path.join(directory, f'{tile}_{band}_{date}.tif'))
