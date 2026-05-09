#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

np.random.seed(0)

root = 'Forest-Change-dataset'
splits = ['train', 'val', 'test']
directories = ['A', 'B', 'label']
N_classes = 2


def create_rgb_image(path: str) -> None:
    arr = np.random.randint(255, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).convert('RGB').save(path)


def create_mask(path: str) -> None:
    arr = np.random.randint(2, size=(32, 32), dtype=np.uint8) * 255
    Image.fromarray(arr).convert('L').save(path)


def create_captions() -> dict:
    return {
        'images': [
            {
                'filename': f'{split}_{i:06d}.png',
                'filepath': split,
                'split': split,
                'sentences': [
                    {'raw': 'minor forest loss is visible'},
                    {'raw': 'small areas of deforestation'},
                    {'raw': 'limited tree cover reduction'},
                    {'raw': 'minor deforestation observed'},
                    {'raw': 'sparse forest loss detected'},
                ],
            }
            for split in splits
            for i in range(N_classes)
        ]
    }


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    root_path = script_dir / root

    if root_path.exists():
        shutil.rmtree(root_path)

    for split in splits:
        for d in directories:
            os.makedirs(root_path / 'images' / split / d)

    for split in splits:
        for i in range(N_classes):
            name = f'{split}_{i:06d}.png'
            create_rgb_image(str(root_path / 'images' / split / 'A' / name))
            create_rgb_image(str(root_path / 'images' / split / 'B' / name))
            create_mask(str(root_path / 'images' / split / 'label' / name))

    with open(root_path / 'ForestChatcaptions.json', 'w') as f:
        json.dump(create_captions(), f)

    sys.path.insert(0, str(Path(__file__).parents[3]))
    from torchgeo.datasets import ForestChange

    ForestChange(root=str(script_dir), split='train')

    zip_path = script_dir / (root + '.zip')
    zip_path.unlink(missing_ok=True)
    shutil.make_archive(
        base_name=str(script_dir / root),
        format='zip',
        root_dir=str(script_dir),
        base_dir=root,
    )
