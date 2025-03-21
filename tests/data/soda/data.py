# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Constants
SPLITS = ['train', 'val', 'test']
IMAGE_SIZE = 32  # Smaller size for tests
NUM_SAMPLES = {'train': 3, 'val': 2, 'test': 2}

CATEGORIES = [
    {'id': 0, 'name': 'airplane'},
    {'id': 1, 'name': 'helicopter'},
    {'id': 2, 'name': 'small-vehicle'},
    {'id': 3, 'name': 'large-vehicle'},
    {'id': 4, 'name': 'ship'},
    {'id': 5, 'name': 'container'},
    {'id': 6, 'name': 'storage-tank'},
    {'id': 7, 'name': 'swimming-pool'},
    {'id': 8, 'name': 'windmill'},
    {'id': 9, 'name': 'ignore'},
]


def create_dummy_image(path: str) -> None:
    """Create a random RGB image."""
    img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_oriented_box() -> list[float]:
    """Create a random oriented bounding box."""
    # Create box center
    cx = np.random.randint(5, IMAGE_SIZE - 5)
    cy = np.random.randint(5, IMAGE_SIZE - 5)

    # Create box dimensions
    w = np.random.randint(5, 10)
    h = np.random.randint(5, 10)

    # Create box corners (clockwise from top-left)
    return [
        float(cx - w / 2),
        float(cy - h / 2),  # top-left
        float(cx + w / 2),
        float(cy - h / 2),  # top-right
        float(cx + w / 2),
        float(cy + h / 2),  # bottom-right
        float(cx - w / 2),
        float(cy + h / 2),  # bottom-left
    ]


def create_annotation(image_id: int, num_objects: int) -> dict[str, Any]:
    """Create annotation file with random objects."""
    data = {
        'type': 'instance',
        'images': {
            'file_name': f'{image_id:05d}.jpg',
            'height': IMAGE_SIZE,
            'width': IMAGE_SIZE,
            'id': image_id,
        },
        'annotations': [],
        'categories': CATEGORIES,
    }

    for i in range(num_objects):
        category_id = np.random.randint(0, len(CATEGORIES) - 1)  # Exclude ignore class
        ann = {
            'poly': create_oriented_box(),
            'area': float(np.random.randint(400, 2000)),
            'category_id': category_id,
            'image_id': image_id,
            'id': i + 1,
        }
        data['annotations'].append(ann)

    return data


def create_dataset() -> None:
    """Create dummy SODA dataset."""
    # Create directory structure
    root = '.'

    samples = []
    image_id = 1

    img_dir = os.path.join(root, 'Images')
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)

    os.makedirs(img_dir, exist_ok=True)

    # Create data for each split
    for split in SPLITS:
        ann_dir = os.path.join(root, 'Annotations', split)

        if os.path.exists(ann_dir):
            shutil.rmtree(ann_dir)

        os.makedirs(ann_dir, exist_ok=True)

        for i in range(NUM_SAMPLES[split]):
            # Create image
            img_name = f'{image_id:05d}.jpg'
            img_path = os.path.join(img_dir, img_name)
            create_dummy_image(img_path)

            # Create annotation
            ann_name = f'{image_id:05d}.json'
            ann_path = os.path.join(ann_dir, ann_name)
            ann_data = create_annotation(image_id, np.random.randint(2, 6))
            with open(ann_path, 'w') as f:
                json.dump(ann_data, f, indent=2)

            # Add to samples
            samples.append(
                {
                    'image_path': os.path.join('Images', img_name),
                    'label_path': os.path.join('Annotations', split, ann_name),
                    'split': split,
                }
            )

            image_id += 1

    # Create sample_df
    df = pd.DataFrame(samples)
    df.to_csv(os.path.join(root, 'sample_df.csv'), index=False)

    # Create archives
    for dirname in ['Images', 'Annotations']:
        archive_name = f'{dirname}.zip'
        archive_path = os.path.join(root, archive_name)

        # Create archive preserving directory structure
        shutil.make_archive(os.path.splitext(archive_path)[0], 'zip', root, dirname)

        # compute md5 hash
        md5 = hashlib.md5()
        with open(archive_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        md5sum = md5.hexdigest()
        print(f'MD5 hash of {archive_path}: {md5sum}')


if __name__ == '__main__':
    create_dataset()
