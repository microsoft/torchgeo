#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# Constants
SIZE = 32  # DIOR uses 800x800 but smaller for tests
CLASSES = [
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'expresswayservicearea',
    'expresswaytollstation',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
]

np.random.seed(0)


def create_image(path: str) -> None:
    """Create random RGB image."""
    img = np.random.randint(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_annotation(path: str, image_name: str) -> None:
    """Create PASCAL VOC annotation file."""
    root = ET.Element('annotation')

    ET.SubElement(root, 'filename').text = image_name

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(SIZE)
    ET.SubElement(size, 'height').text = str(SIZE)
    ET.SubElement(size, 'depth').text = '3'

    # Add 1-3 random objects
    for _ in range(np.random.randint(1, 4)):
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = np.random.choice(CLASSES)

        # Create random box coordinates
        x1 = np.random.randint(0, SIZE // 2)
        y1 = np.random.randint(0, SIZE // 2)
        x2 = np.random.randint(x1 + SIZE // 4, SIZE)
        y2 = np.random.randint(y1 + SIZE // 4, SIZE)

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(x1)
        ET.SubElement(bbox, 'ymin').text = str(y1)
        ET.SubElement(bbox, 'xmax').text = str(x2)
        ET.SubElement(bbox, 'ymax').text = str(y2)

    tree = ET.ElementTree(root)
    tree.write(path)


def create_dataset():
    """Create dummy DIOR dataset."""
    root = os.getcwd()

    img_dir = os.path.join(root, 'Images')
    ann_dir = os.path.join(root, 'Annotations')

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if os.path.exists(ann_dir):
        shutil.rmtree(ann_dir)

    # Create directories
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for split in ['trainval', 'test']:
        os.makedirs(os.path.join(img_dir, split), exist_ok=True)
        if split == 'trainval':
            os.makedirs(os.path.join(ann_dir, split), exist_ok=True)

    samples = []

    # Create trainval data
    for idx in range(6):
        img_name = f'{idx:06d}.jpg'
        ann_name = f'{idx:06d}.xml'

        # Create files
        create_image(os.path.join(root, 'Images', 'trainval', img_name))
        create_annotation(
            os.path.join(root, 'Annotations', 'trainval', ann_name), img_name
        )

        # Add to samples
        split = 'train' if idx < 4 else 'val'
        samples.append(
            {
                'image_path': os.path.join('Images', 'trainval', img_name),
                'label_path': os.path.join('Annotations', 'trainval', ann_name),
                'split': split,
            }
        )

    # Create test data (2 samples)
    for idx in range(2):
        img_name = f'{idx:06d}.jpg'
        create_image(os.path.join(root, 'Images', 'test', img_name))
        samples.append(
            {
                'image_path': os.path.join('Images', 'test', img_name),
                'label_path': None,  # No annotations for test
                'split': 'test',
            }
        )

    df = pd.DataFrame(samples)
    df.to_parquet(os.path.join('sample_df.parquet'))

    for dirname in ['Images', 'Annotations']:
        archive_name = f'{dirname}_trainval.zip'
        archive_path = os.path.join(root, archive_name)

        shutil.make_archive(
            archive_path.split('.')[0],
            'zip',
            os.path.join(root, dirname, '..'),
            os.path.join(dirname, 'trainval'),
        )

        with open(archive_path, 'rb') as archive_file:
            md5 = hashlib.md5(archive_file.read()).hexdigest()
        print(f'{archive_name}: {md5}')

    archive_name = 'Images_test.zip'
    archive_path = os.path.join(root, archive_name)

    shutil.make_archive(
        archive_path.split('.')[0],
        'zip',
        os.path.join(root, 'Images', '..'),
        os.path.join('Images', 'test'),
    )

    with open(archive_path, 'rb') as archive_file:
        md5 = hashlib.md5(archive_file.read()).hexdigest()
    print(f'{archive_name}: {md5}')


if __name__ == '__main__':
    create_dataset()
