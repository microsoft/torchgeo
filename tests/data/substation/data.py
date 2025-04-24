#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import zipfile
from typing import Literal

import numpy as np

# Parameters
SIZE = 32  # Image dimensions
NUM_SAMPLES = 5  # Number of samples
np.random.seed(0)

# Define directory hierarchy
FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]

filenames: FILENAME_HIERARCHY = {'image_stack': ['image'], 'mask': ['mask']}


def create_file(path: str, value: Literal['image', 'mask']) -> None:
    """Generates .npz files for images or masks based on the path.

    Args:
        path: Base path for saving files.
        value: Type of file, either 'image' or 'mask'.
    """
    for i in range(NUM_SAMPLES):
        new_path = f'{path}_{i}.npz'

        if value == 'image':
            # Generate image data with shape (4, 13, SIZE, SIZE) for timepoints and channels
            data = np.random.rand(4, 13, SIZE, SIZE).astype(np.float32)
        elif value == 'mask':
            # Generate mask data with shape (SIZE, SIZE) with 4 classes
            data = np.random.randint(0, 4, size=(SIZE, SIZE)).astype(np.uint8)

        np.savez_compressed(new_path, arr_0=data)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    """
    Recursively creates directory structure based on hierarchy and populates with data files.

    Args:
        directory: Base directory for dataset.
        hierarchy: Directory and file structure.
    """
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, 'image')
            create_file(path, value)


if __name__ == '__main__':
    # Generate directory structure and data
    create_directory('.', filenames)

    # Create zip archives of dataset folders
    filename_images = 'image_stack.tar.gz'
    filename_masks = 'mask.tar.gz'
    shutil.make_archive('image_stack', 'gztar', '.', 'image_stack')
    shutil.make_archive('mask', 'gztar', '.', 'mask')

    # Compute and print MD5 checksums for data validation
    with open(filename_images, 'rb') as f:
        md5_images = hashlib.md5(f.read()).hexdigest()
        print(f'{filename_images}: {md5_images}')

    with open(filename_masks, 'rb') as f:
        md5_masks = hashlib.md5(f.read()).hexdigest()
        print(f'{filename_masks}: {md5_masks}')
