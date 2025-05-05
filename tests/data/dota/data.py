# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def create_dummy_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create small dummy image."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_annotation_file(
    path: Path, is_hbb: bool = False, no_boxes: bool = False
) -> None:
    """Create dummy annotation file with scaled coordinates."""
    if is_hbb:
        # Horizontal boxes scaled for 64x64
        boxes = [
            '10.0 10.0 20.0 10.0 20.0 20.0 10.0 20.0 plane 0\n',
            '30.0 30.0 40.0 30.0 40.0 40.0 30.0 40.0 ship 0\n',
        ]
    else:
        # Oriented boxes scaled for 64x64
        boxes = [
            '10.0 10.0 20.0 12.0 18.0 20.0 8.0 18.0 plane 0\n',
            '30.0 30.0 42.0 32.0 40.0 40.0 28.0 38.0 ship 0\n',
        ]

    if no_boxes:
        boxes = []

    with open(path, 'w') as f:
        f.write('imagesource:dummy\n')
        f.write('gsd:1.0\n')
        f.writelines(boxes)


def create_test_data(root: Path) -> None:
    """Create DOTA test dataset."""
    splits = ['train', 'val']
    versions = ['1.0', '1.5', '2.0']

    # Create directory structure
    for split in splits:
        num_samples = 3 if split == 'train' else 2

        if os.path.exists(root / split):
            shutil.rmtree(root / split)
        for version in versions:
            # Create images and annotations
            for i in range(num_samples):
                img_name = f'P{i:04d}.png'
                ann_name = f'P{i:04d}.txt'

                # Create directories
                (root / split / 'images').mkdir(parents=True, exist_ok=True)
                (root / split / 'annotations' / f'version{version}').mkdir(
                    parents=True, exist_ok=True
                )

                # Create files
                if i == 0:
                    no_boxes = True
                else:
                    no_boxes = False
                create_dummy_image(root / split / 'images' / img_name)
                create_annotation_file(
                    root / split / 'annotations' / f'version{version}' / ann_name,
                    False,
                    no_boxes,
                )

            # Create tar archives
            for type_ in ['images', 'annotations']:
                src_dir = root / split / type_
                if src_dir.exists():
                    tar_name = f'dotav{version}_{type_}_{split}.tar.gz'
                    with tarfile.open(root / tar_name, 'w:gz') as tar:
                        tar.add(src_dir, arcname=f'{split}/{type_}')

            # print md5sums
            def md5(fname: str) -> str:
                hash_md5 = hashlib.md5()
                with open(fname, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()

    print('file_info = {')
    for split in splits:
        print(f"    '{split}': {{")

        for type_ in ['images', 'annotations']:
            print(f"        '{type_}': {{")

            for version in versions:
                tar_name = f'dotav{version}_{type_}_{split}.tar.gz'
                checksum = md5(tar_name)

                # version 1.0 and 1.5 have the same images
                if version == '1.5' and type_ == 'images':
                    version_filename = '1.0'
                else:
                    version_filename = version

                print(f"            '{version}': {{")
                print(
                    f"                'filename': 'dotav{version_filename}_{type_}_{split}.tar.gz',"
                )
                print(f"                'md5': '{checksum}',")
                print('            },')

            print('        },')

        print('    },')
    print('}')


def create_sample_df(root: Path) -> pd.DataFrame:
    """Create sample DataFrame for test data."""
    rows = []
    splits = ['train', 'val']
    versions = ['1.0', '1.5', '2.0']

    for split in splits:
        num_samples = 3 if split == 'train' else 2
        for version in versions:
            for i in range(num_samples):
                img_name = f'P{i:04d}.png'
                ann_name = f'P{i:04d}.txt'

                row = {
                    'image_path': str(Path(split) / 'images' / img_name),
                    'annotation_path': str(
                        Path(split) / 'annotations' / f'version{version}' / ann_name
                    ),
                    'split': split,
                    'version': version,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(root / 'samples.csv')
    return df


if __name__ == '__main__':
    root = Path('.')
    create_test_data(root)
    df = create_sample_df(root)
