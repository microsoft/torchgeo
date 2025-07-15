#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
from PIL import Image

np.random.seed(0)


def create_image(path: str) -> None:
    Z = np.random.randint(255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(Z).convert('RGB')
    img.save(path)


def create_mask(path: str) -> None:
    Z = np.random.randint(2, size=(32, 32, 3), dtype=np.uint8) * 255
    img = Image.fromarray(Z).convert('L')
    img.save(path)


if __name__ == '__main__':
    root = 'LEVIR-CD+'
    splits = ['train', 'test']
    directories = ['A', 'B', 'label']

    if os.path.exists(root):
        shutil.rmtree(root)

    for split in splits:
        for directory in directories:
            os.makedirs(os.path.join(root, split, directory))

        for i in range(2):
            folder = os.path.join(root, split, 'A')
            path = os.path.join(folder, f'0{i}.png')
            create_image(path)

            folder = os.path.join(root, split, 'B')
            path = os.path.join(folder, f'0{i}.png')
            create_image(path)

            folder = os.path.join(root, split, 'label')
            path = os.path.join(folder, f'0{i}.png')
            create_mask(path)

    # Compress data
    shutil.make_archive(root, 'zip', '.', root)

    # compute checksum
    with open(f'{root}.zip', 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{root}.zip: {md5}')
