#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import zipfile
from typing import Literal

import numpy as np

# Parameters
SIZE = 32  # Image dimensions
NUM_SAMPLES = 5  # Number of samples
NUM_PARTS = 3  # Number of parts of the images archive
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

    # The real images live in an 'images' directory of a multi-part zip archive
    with zipfile.ZipFile('images.zip', 'w', zipfile.ZIP_DEFLATED) as f:
        f.mkdir('images')
        for name in sorted(os.listdir('image_stack')):
            f.write(os.path.join('image_stack', name), os.path.join('images', name))

    # Split the archive the way 'zip -s' does: a spanning signature followed by the
    # archive itself, cut at fixed-size boundaries. The central directory offsets are
    # not rewritten to be relative to each part, but the dataset never reads them.
    with open('images.zip', 'rb') as f:
        data = b'PK\x07\x08' + f.read()

    chunk_size = len(data) // NUM_PARTS + 1
    for i in range(NUM_PARTS - 1):
        with open(f'images.z{i + 1:02}', 'wb') as f:
            f.write(data[i * chunk_size : (i + 1) * chunk_size])

    with open('images.zip', 'wb') as f:
        f.write(data[(NUM_PARTS - 1) * chunk_size :])

    shutil.make_archive('mask', 'gztar', '.', 'mask')
