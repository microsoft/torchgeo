#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import pandas as pd
from PIL import Image

DTYPE = np.uint8
SIZE = 2

np.random.seed(0)

for split in ['train', 'test']:
    os.makedirs(split, exist_ok=True)

    filename = split
    if split == 'train':
        filename = 'training'

    features = pd.read_csv(f'{filename}_set_features.csv')
    for image_id, _, _, ocean in features.values:
        size = (SIZE, SIZE)
        if ocean % 2 == 0:
            size = (SIZE * 2, SIZE * 2, 3)

        arr = np.random.randint(np.iinfo(DTYPE).max, size=size, dtype=DTYPE)
        img = Image.fromarray(arr)
        img.save(os.path.join(split, f'{image_id}.jpg'))
