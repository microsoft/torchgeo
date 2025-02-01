# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import os
from pathlib import Path
import tarfile
import pandas as pd
from PIL import Image
import hashlib
import shutil


def create_dummy_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create small dummy image."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_annotation_file(path: Path, is_hbb: bool = False, no_boxes=False) -> None:
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
    versions = ['1.0', '2.0']

    # Create directory structure
    for split in splits:
        num_samples = 3 if split == 'train' else 2

        if os.path.exists(root / split):
            shutil.rmtree(root / split)
        for version in versions:
            # Create images and annotations
            for i in range(num_samples):
                img_name = f'P{version[0]}_{i:04d}.png'
                ann_name = f'P{version[0]}_{i:04d}.txt'

                # Create directories
                (root / split / 'images').mkdir(parents=True, exist_ok=True)
                (root / split / 'annotations').mkdir(parents=True, exist_ok=True)
                if version == '2.0':
                    (root / split / 'annotations_hbb').mkdir(
                        parents=True, exist_ok=True
                    )

                # Create files
                if i == 0:
                    no_boxes = True
                else:
                    no_boxes = False
                create_dummy_image(root / split / 'images' / img_name)
                create_annotation_file(
                    root / split / 'annotations' / ann_name, False, no_boxes
                )
                if version == '2.0':
                    create_annotation_file(
                        root / split / 'annotations_hbb' / ann_name, True, no_boxes
                    )

            # Create tar archives
            for type_ in ['images', 'annotations']:
                src_dir = root / split / type_
                if src_dir.exists():
                    tar_name = f'dotav{version[0]}_{type_}_{split}.tar.gz'
                    with tarfile.open(root / tar_name, 'w:gz') as tar:
                        tar.add(src_dir, arcname=f'{split}/{type_}')

            # print md5sums
            def md5(fname: str) -> str:
                hash_md5 = hashlib.md5()
                with open(fname, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()

    for version in versions:
        for type_ in ['images', 'annotations']:
            for split in splits:
                tar_name = f'dotav{version[0]}_{type_}_{split}.tar.gz'
                print(f'{tar_name} md5: {md5(tar_name)}')


def create_sample_df(root: Path) -> pd.DataFrame:
    """Create sample DataFrame for test data."""
    rows = []
    splits = ['train', 'val']
    versions = ['1.0', '2.0']

    for split in splits:
        num_samples = 3 if split == 'train' else 2
        for version in versions:
            for i in range(num_samples):
                img_name = f'P{version[0]}_{i:04d}.png'
                ann_name = f'P{version[0]}_{i:04d}.txt'

                row = {
                    'image_path': str(Path(split) / 'images' / img_name),
                    'annotation_path': str(Path(split) / 'annotations' / ann_name),
                    'split': split,
                    'version': version,
                }

                if version == '2.0':
                    row['annotation_hbb_path'] = str(
                        Path(split) / 'annotations_hbb' / ann_name
                    )
                else:
                    row['annotation_hbb_path'] = None

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(root / 'samples.parquet')
    return df


if __name__ == '__main__':
    root = Path('.')
    create_test_data(root)
    df = create_sample_df(root)
