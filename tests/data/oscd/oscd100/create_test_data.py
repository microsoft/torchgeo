#!/usr/bin/env python3
"""Generate minimal test data for OSCD100 in GeoTIFF format."""

import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# Create test structure
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

# Minimal 8x8 test images
SIZE = 8

# Create directories
base = Path(__file__).parent
images_dir = base / 'OSCD100_Images'
images_dir.mkdir(exist_ok=True)

# Create 2 crops per split
for split in splits:
    labels_dir = base / f'OSCD100_{split.capitalize()}_Labels'
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

        # Labels
        crop_label_dir = labels_dir / crop_name / 'cm'
        crop_label_dir.mkdir(parents=True, exist_ok=True)

        # Random binary mask
        mask = np.random.randint(0, 2, (SIZE, SIZE), dtype=np.uint8) * 255
        Image.fromarray(mask, mode='L').save(crop_label_dir / 'cm.png')

print('Creating zip archives...')

# Zip Images
with zipfile.ZipFile(base / 'oscd100_images.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(base)
            zf.write(file_path, arcname)

# Zip each split's labels
for split in splits:
    labels_dir = base / f'OSCD100_{split.capitalize()}_Labels'
    with zipfile.ZipFile(
        base / f'oscd100_{split}_labels.zip', 'w', zipfile.ZIP_DEFLATED
    ) as zf:
        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(base)
                zf.write(file_path, arcname)

print('Test data created!')
