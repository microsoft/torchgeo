#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 8
np.random.seed(0)

splits = ['train', 'val', 'test']
bands = [
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
    'B10',
    'B11',
    'B12',
]

# Clean up old directories
for dirname in [
    'Onera Satellite Change Detection dataset - Images',
    'Onera Satellite Change Detection dataset - Train Labels',
    'Onera Satellite Change Detection dataset - Val Labels',
    'Onera Satellite Change Detection dataset - Test Labels',
]:
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

for split in splits:
    fname = (
        f'Onera Satellite Change Detection dataset - {split.capitalize()} Labels.zip'
    )
    if os.path.exists(fname):
        os.remove(fname)

fname = 'Onera Satellite Change Detection dataset - Images.zip'
if os.path.exists(fname):
    os.remove(fname)

# Create directories
images_dir = Path('Onera Satellite Change Detection dataset - Images')
images_dir.mkdir(exist_ok=True)

# Create 2 crops per split
for split in splits:
    labels_dir = Path(
        f'Onera Satellite Change Detection dataset - {split.capitalize()} Labels'
    )
    labels_dir.mkdir(exist_ok=True)

    for i in range(2):
        crop_name = f'{split}_{i:03d}'

        # Images
        crop_img_dir = images_dir / crop_name
        imgs1_dir = crop_img_dir / 'imgs_1_rect'
        imgs2_dir = crop_img_dir / 'imgs_2_rect'
        imgs1_dir.mkdir(parents=True, exist_ok=True)
        imgs2_dir.mkdir(parents=True, exist_ok=True)

        for band in bands:
            # Random uint16 data for each band
            data1 = np.random.randint(0, 10000, (SIZE, SIZE), dtype=np.uint16)
            data2 = np.random.randint(0, 10000, (SIZE, SIZE), dtype=np.uint16)

            Image.fromarray(data1).save(imgs1_dir / f'{band}.tif')
            Image.fromarray(data2).save(imgs2_dir / f'{band}.tif')

        # Create dates.txt for OSCD compatibility
        dates_file = crop_img_dir / 'dates.txt'
        dates_file.write_text('img1 2020-01-01\nimg2 2020-06-01\n')

        # Labels
        crop_label_dir = labels_dir / crop_name / 'cm'
        crop_label_dir.mkdir(parents=True, exist_ok=True)

        # Random binary mask
        mask = np.random.randint(0, 2, (SIZE, SIZE), dtype=np.uint8) * 255
        Image.fromarray(mask, mode='L').save(crop_label_dir / 'cm.png')

# Zip Images
with zipfile.ZipFile(
    'Onera Satellite Change Detection dataset - Images.zip', 'w', zipfile.ZIP_DEFLATED
) as zf:
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            file_path = Path(root) / file
            zf.write(file_path, file_path)

# Zip each split's labels
for split in splits:
    labels_dir = Path(
        f'Onera Satellite Change Detection dataset - {split.capitalize()} Labels'
    )
    filename = (
        f'Onera Satellite Change Detection dataset - {split.capitalize()} Labels.zip'
    )
    with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                file_path = Path(root) / file
                zf.write(file_path, file_path)

# Print MD5s for updating test file
with open('Onera Satellite Change Detection dataset - Images.zip', 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(
        repr('Onera Satellite Change Detection dataset - Images.zip')
        + ': '
        + repr(md5)
        + ','
    )

for split in splits:
    filename = (
        f'Onera Satellite Change Detection dataset - {split.capitalize()} Labels.zip'
    )
    with open(filename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(repr(filename) + ': ' + repr(md5) + ',')
