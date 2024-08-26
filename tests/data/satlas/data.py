#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil

from PIL import Image

SIZE = 32

index = [[7149, 3246], [1234, 5678]]

FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]
filenames: FILENAME_HIERARCHY = {
    'landsat': {'2022-03': list(f'b{i}' for i in range(1, 12))},
    'naip': {'m_3808245_se_17_1_20110801': ['tci', 'ir']},
    'sentinel1': {'2022-01': ['vh', 'vv']},
    'sentinel2': {
        'S2A_MSIL1C_20220309T032601_N0400_R018_T48RYR_20220309T060235': [
            'tci',
            'b05',
            'b06',
            'b07',
            'b08',
            'b11',
            'b12',
        ]
    },
}


def create_files(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for col, row in index:
        mode = 'RGB' if path.endswith('tci') else 'L'
        img = Image.new(mode, (SIZE, SIZE))
        img.save(os.path.join(path, f'{col}_{row}.png'))


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_files(path)


if __name__ == '__main__':
    create_directory('.', filenames)

    col, row = index[0]
    path = os.path.join('static', f'{col}_{row}')
    os.makedirs(path, exist_ok=True)
    img = Image.new('L', (SIZE, SIZE))
    img.save(os.path.join(path, 'land_cover.png'))

    os.makedirs('metadata', exist_ok=True)
    with open(os.path.join('metadata', 'train_lowres.json'), 'w') as f:
        json.dump(index, f)

    for path in os.listdir('.'):
        if os.path.isdir(path):
            shutil.make_archive(path, 'tar', '.', path)
