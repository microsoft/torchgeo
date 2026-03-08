#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import csv
import os
import shutil

import numpy as np
from PIL import Image
from torchvision.datasets.utils import calculate_md5


def generate_test_data(root: str) -> str:
    """Create test data archive for DLRSD dataset.

    Args:
        root: path to store test data

    Returns:
        md5 hash of created archive
    """
    dtype = np.uint8
    size = 4
    num_images = 2
    scene_classes = ('agricultural', 'airplane')

    folder_path = os.path.join(root, 'DLRSD')
    images_dir = os.path.join(folder_path, 'Images')
    labels_dir = os.path.join(folder_path, 'Labels')

    # Multilabel CSV data (columns: image, airplane, baresoil, ..., water)
    csv_rows = []

    for cls in scene_classes:
        os.makedirs(os.path.join(images_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, cls), exist_ok=True)

        for i in range(num_images):
            name = f'{cls}{i:02d}'

            # Create RGB TIFF image
            arr = np.random.randint(0, 256, (size, size, 3), dtype=dtype)
            img = Image.fromarray(arr, mode='RGB')
            img.save(os.path.join(images_dir, cls, f'{name}.tif'))

            # Create palette PNG mask with values 1-17
            mask_val = (i % 17) + 1
            mask_arr = np.full((size, size), mask_val, dtype=dtype)
            mask_img = Image.fromarray(mask_arr)
            mask_img = mask_img.convert('P')
            mask_img.save(os.path.join(labels_dir, cls, f'{name}.png'))

            # Create multilabel row (binary vector of length 17)
            labels = [0] * 17
            labels[mask_val - 1] = 1
            csv_rows.append([name, *labels])

    # Write multilabels CSV
    header = [
        'image',
        'airplane',
        'baresoil',
        'buildings',
        'cars',
        'chaparral',
        'court',
        'dock',
        'field',
        'grass',
        'mobilehome',
        'pavement',
        'sand',
        'sea',
        'ship',
        'tanks',
        'trees',
        'water',
    ]
    csv_path = os.path.join(folder_path, 'multilabels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    # Create archive
    shutil.make_archive(os.path.join(root, 'DLRSD'), 'zip', root, 'DLRSD')
    shutil.rmtree(folder_path)
    return calculate_md5(os.path.join(root, 'DLRSD.zip'))


if __name__ == '__main__':
    md5_hash = generate_test_data(os.getcwd())
    print(md5_hash + '\n')
