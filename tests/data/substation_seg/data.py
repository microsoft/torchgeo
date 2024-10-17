#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np

SIZE = 228
NUM_SAMPLES = 5
np.random.seed(0)

FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]

filenames: FILENAME_HIERARCHY = {
    'image_stack': ['image'],
    'mask': ['mask'],
}

def create_file(path: str) -> None:
    for i in range(NUM_SAMPLES):
        new_path = f'{path}_{i}.npz'
        fn = os.path.basename(new_path)
        if fn.startswith('image'):
            data = np.random.rand(4, SIZE, SIZE).astype(np.float32)  # 4 channels (RGB + NIR)
        elif fn.startswith('mask'):
            data = np.random.randint(0, 4, size=(SIZE, SIZE)).astype(np.uint8)  # Mask with 4 classes
        np.savez_compressed(new_path, arr_0=data)

def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path)

if __name__ == '__main__':
    create_directory('.', filenames)

    # Create a zip archive of the generated dataset
    filename_images = 'image_stack.tar.gz'
    filename_masks = 'mask.tar.gz'
    shutil.make_archive('image_stack', 'gztar', '.', 'image_stack')
    shutil.make_archive('mask', 'gztar', '.', 'mask')

    # Compute checksums
    with open(filename_images, 'rb') as f:
        md5_images = hashlib.md5(f.read()).hexdigest()
        print(f'{filename_images}: {md5_images}')
    
    with open(filename_masks, 'rb') as f:
        md5_masks = hashlib.md5(f.read()).hexdigest()
        print(f'{filename_masks}: {md5_masks}')
