#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
from PIL import Image

SIZE = 4
np.random.seed(0)

splits = ['train', 'val', 'test']
directories = ['A', 'B', 'label']

for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

for split in splits:
    filename = f'oscd100_{split}.zip'
    if os.path.exists(filename):
        os.remove(filename)

for split in splits:
    for i in range(2):
        for directory in ['A', 'B']:
            filename = os.path.join(directory, f'{split}_{i:03d}.png')
            arr = np.random.randint(256, size=(SIZE, SIZE, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode='RGB')
            img.save(filename)

        filename = os.path.join('label', f'{split}_{i:03d}.png')
        arr = np.random.randint(2, size=(SIZE, SIZE), dtype=np.uint8) * 255
        img = Image.fromarray(arr, mode='L')
        img.save(filename)

for split in splits:
    filename = f'oscd100_{split}.zip'
    files_to_zip = []
    for directory in directories:
        for f in os.listdir(directory):
            if f.startswith(split):
                files_to_zip.append(os.path.join(directory, f))

    import zipfile

    with zipfile.ZipFile(filename, 'w') as zf:
        for f in files_to_zip:
            zf.write(f)

    with open(filename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(repr(filename) + ': ' + repr(md5) + ',')
