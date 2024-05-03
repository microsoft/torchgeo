#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil

import h5py
import numpy as np

SIZE = 32

np.random.seed(0)

PATHS = [
    os.path.join('cropharvest', 'features', 'arrays', '0_TestDataset1.h5'),
    os.path.join('cropharvest', 'features', 'arrays', '1_TestDataset1.h5'),
    os.path.join('cropharvest', 'features', 'arrays', '2_TestDataset1.h5'),
    os.path.join('cropharvest', 'features', 'arrays', '0_TestDataset2.h5'),
    os.path.join('cropharvest', 'features', 'arrays', '1_TestDataset2.h5'),
]


def create_geojson():
    geojson = {
        'type': 'FeatureCollection',
        'crs': {},
        'features': [
            {
                'type': 'Feature',
                'properties': {
                    'dataset': 'TestDataset1',
                    'index': 0,
                    'is_crop': 1,
                    'label': 'soybean',
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            },
            {
                'type': 'Feature',
                'properties': {
                    'dataset': 'TestDataset1',
                    'index': 0,
                    'is_crop': 1,
                    'label': 'alfalfa',
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            },
            {
                'type': 'Feature',
                'properties': {
                    'dataset': 'TestDataset1',
                    'index': 1,
                    'is_crop': 1,
                    'label': None,
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            },
            {
                'type': 'Feature',
                'properties': {
                    'dataset': 'TestDataset2',
                    'index': 2,
                    'is_crop': 1,
                    'label': 'maize',
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            },
            {
                'type': 'Feature',
                'properties': {
                    'dataset': 'TestDataset2',
                    'index': 1,
                    'is_crop': 0,
                    'label': None,
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            },
        ],
    }
    return geojson


def create_file(path: str) -> None:
    Z = np.random.randint(4000, size=(12, 18), dtype=np.int64)
    with h5py.File(path, 'w') as f:
        f.create_dataset('array', data=Z)


if __name__ == '__main__':
    directory = 'cropharvest'

    # remove old data
    to_remove = [
        os.path.join(directory, 'features'),
        os.path.join(directory, 'features.tar.gz'),
        os.path.join(directory, 'labels.geojson'),
    ]
    for path in to_remove:
        if os.path.isdir(path):
            shutil.rmtree(path)

    label_path = os.path.join(directory, 'labels.geojson')
    geojson = create_geojson()
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, 'w') as f:
        json.dump(geojson, f)

    for path in PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        create_file(path)

    # compress data
    source_dir = os.path.join(directory, 'features')
    shutil.make_archive(source_dir, 'gztar', directory, 'features')

    # compute checksum
    with open(label_path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{label_path}: {md5}')

    with open(os.path.join(directory, 'features.tar.gz'), 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'zipped features: {md5}')
