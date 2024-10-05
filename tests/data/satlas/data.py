#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil

from PIL import Image

SIZE = 32
landsat_size = {
    'b1': SIZE // 2,
    'b2': SIZE // 2,
    'b3': SIZE // 2,
    'b4': SIZE // 2,
    'b5': SIZE // 2,
    'b6': SIZE // 2,
    'b7': SIZE // 2,
    'b8': SIZE,
    'b9': SIZE // 2,
    'b10': SIZE // 2,
    'b11': SIZE // 4,
    'b12': SIZE // 4,
}

index = [[7149, 3246], [1234, 5678]]
good_images = [
    [7149, 3246, '2022-03'],
    [1234, 5678, '2022-03'],
    [7149, 3246, 'm_3808245_se_17_1_20110801'],
    [1234, 5678, 'm_3808245_se_17_1_20110801'],
    [7149, 3246, '2022-01'],
    [1234, 5678, '2022-01'],
    [7149, 3246, 'S2A_MSIL1C_20220309T032601_N0400_R018_T48RYR_20220309T060235'],
    [1234, 5678, 'S2A_MSIL1C_20220309T032601_N0400_R018_T48RYR_20220309T060235'],
]
times = {
    '2022-03': '2022-03-01T00:00:00+00:00',
    'm_3808245_se_17_1_20110801': '2011-08-01T12:00:00+00:00',
    '2022-01': '2022-01-01T00:00:00+00:00',
    'S2A_MSIL1C_20220309T032601_N0400_R018_T48RYR_20220309T060235': '2022-03-09T06:02:35+00:00',
}

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
        band = os.path.basename(path)
        mode = 'RGB' if band == 'tci' else 'L'
        size = SIZE
        if 'landsat' in path:
            size = landsat_size[band]
        img = Image.new(mode, (size, size))
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

    with open(os.path.join('metadata', 'good_images_lowres_all.json'), 'w') as f:
        json.dump(good_images, f)

    with open(os.path.join('metadata', 'image_times.json'), 'w') as f:
        json.dump(times, f)

    for path in os.listdir('.'):
        if os.path.isdir(path):
            shutil.make_archive(path, 'tar', '.', path)
