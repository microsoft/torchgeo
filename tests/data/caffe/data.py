#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
from PIL import Image

# Define the root directory and subdirectories
root_dir = 'caffe'
sub_dirs = ['zones', 'sar_images', 'fronts']
splits = ['train', 'val', 'test']

zone_file_names = [
    'Crane_2002-11-09_ERS_20_2_061_zones__93_102_0_0_0.png',
    'Crane_2007-09-22_ENVISAT_20_1_467_zones__93_102_8_1024_0.png',
    'JAC_2015-12-23_TSX_6_1_005_zones__57_49_195_384_1024.png',
]

IMG_SIZE = 32


# Function to create dummy images
def create_dummy_image(path: str, shape: tuple[int], pixel_values: list[int]) -> None:
    data = np.random.choice(pixel_values, size=shape, replace=True).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(path)


def create_zone_images(split: str, filename: str) -> None:
    zone_pixel_values = [0, 64, 127, 254]
    path = os.path.join(root_dir, 'zones', split, filename)
    create_dummy_image(path, (IMG_SIZE, IMG_SIZE), zone_pixel_values)


def create_sar_images(split: str, filename: str) -> None:
    sar_pixel_values = range(256)
    path = os.path.join(root_dir, 'sar_images', split, filename)
    create_dummy_image(path, (IMG_SIZE, IMG_SIZE), sar_pixel_values)


def create_front_images(split: str, filename: str) -> None:
    front_pixel_values = [0, 255]
    path = os.path.join(root_dir, 'fronts', split, filename)
    create_dummy_image(path, (IMG_SIZE, IMG_SIZE), front_pixel_values)


if os.path.exists(root_dir):
    shutil.rmtree(root_dir)

# Create the directory structure
for sub_dir in sub_dirs:
    for split in splits:
        os.makedirs(os.path.join(root_dir, sub_dir, split), exist_ok=True)

# Create dummy data for all splits and filenames
for split in splits:
    for filename in zone_file_names:
        create_zone_images(split, filename)
        create_sar_images(split, filename.replace('_zones_', '_'))
        create_front_images(split, filename.replace('_zones_', '_front_'))

# zip and compute md5
shutil.make_archive(root_dir, 'zip', '.', root_dir)


def md5(fname: str) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


md5sum = md5('caffe.zip')
print(f'MD5 checksum: {md5sum}')
