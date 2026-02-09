# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def create_dummy_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create dummy RGB image."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_coco_annotations(split: str, num_images: int) -> dict:
    """Create COCO format annotations."""
    return {
        'info': {'year': 2023, 'version': '1.0'},
        'images': [
            {'id': i, 'file_name': f'P{i:04d}.png', 'height': 64, 'width': 64}
            for i in range(num_images)
        ],
        'annotations': [
            {
                'id': i,
                'image_id': i // 2,  # 2 annotations per image
                'category_id': i % 15,
                'segmentation': [[10, 10, 20, 10, 20, 20, 10, 20]],
                'area': 100,
                'bbox': [10, 10, 10, 10],
                'iscrowd': 0,
            }
            for i in range(num_images * 2)
        ],
        'categories': [
            {'id': i, 'name': name}
            for i, name in enumerate(
                [
                    'plane',
                    'ship',
                    'storage tank',
                    'baseball diamond',
                    'tennis court',
                    'basketball court',
                    'ground track field',
                    'harbor',
                    'bridge',
                    'vehicle',
                    'helicopter',
                    'roundabout',
                    'swimming pool',
                    'soccer ball field',
                    'container crane',
                ]
            )
        ],
    }


def create_test_data(root: Path) -> None:
    """Create iSAID test dataset."""
    splits = {'train': 3, 'val': 2}

    for split, num_samples in splits.items():
        if os.path.exists(root / split):
            shutil.rmtree(root / split)

        # Create directories
        for subdir in ['images', 'Annotations', 'Instance_masks', 'Semantic_masks']:
            (root / split / subdir).mkdir(parents=True, exist_ok=True)

        # Create images and masks
        for i in range(num_samples):
            # RGB image
            create_dummy_image(root / split / 'images' / f'P{i:04d}.png')

            # Instance mask (R+G*256+B*256^2 encoding)
            instance_mask = np.zeros((64, 64, 3), dtype=np.uint8)
            instance_mask[10:20, 10:20, 0] = i + 1  # R channel for unique IDs
            Image.fromarray(instance_mask).save(
                root / split / 'Instance_masks' / f'P{i:04d}.png'
            )

            # Semantic mask (similar encoding for class IDs)
            semantic_mask = np.zeros((64, 64, 3), dtype=np.uint8)
            semantic_mask[10:20, 10:20, 0] = 1  # Class ID 1
            Image.fromarray(semantic_mask).save(
                root / split / 'Semantic_masks' / f'P{i:04d}.png'
            )

        # Create COCO annotations
        annotations = create_coco_annotations(split, num_samples)
        with open(root / split / 'Annotations' / f'iSAID_{split}.json', 'w') as f:
            json.dump(annotations, f)

        # Create image tar
        img_tar = f'dotav1_images_{split}.tar.gz'
        with tarfile.open(root / img_tar, 'w:gz') as tar:
            tar.add(root / split / 'images', arcname=os.path.join(split, 'images'))

        # Create annotations tar with all splits
        ann_tar = f'isaid_annotations_{split}.tar.gz'
        with tarfile.open(root / ann_tar, 'w:gz') as tar:
            for split in splits:
                for subdir in ['Annotations', 'Instance_masks', 'Semantic_masks']:
                    tar.add(root / split / subdir, arcname=os.path.join(split, subdir))

    # print md5sums
    def md5(fname: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fname, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    # Print MD5 checksums
    for split in splits:
        print(
            f'MD5 for dotav1_images_{split}.tar.gz: '
            f'{md5(root / f"dotav1_images_{split}.tar.gz")}'
        )
        print(
            f'MD5 for isaid_annotations_{split}.tar.gz: {md5(root / f"isaid_annotations_{split}.tar.gz")}'
        )


if __name__ == '__main__':
    root = Path('.')
    create_test_data(root)
