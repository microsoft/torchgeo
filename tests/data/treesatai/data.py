#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import json
import os
import random
import shutil
import zipfile

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32

random.seed(0)
np.random.seed(0)

classes = (
    'Abies',
    'Acer',
    'Alnus',
    'Betula',
    'Cleared',
    'Fagus',
    'Fraxinus',
    'Larix',
    'Picea',
    'Pinus',
    'Populus',
    'Prunus',
    'Pseudotsuga',
    'Quercus',
    'Tilia',
)

species = (
    'Acer_pseudoplatanus',
    'Alnus_spec',
    'Fagus_sylvatica',
    'Picea_abies',
    'Pseudotsuga_menziesii',
    'Quercus_petraea',
    'Quercus_rubra',
)

profile = {
    'aerial': {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': None,
        'width': SIZE,
        'height': SIZE,
        'count': 4,
        'crs': CRS.from_epsg(25832),
        'transform': Affine(
            0.19999999999977022, 0.0, 552245.4, 0.0, -0.19999999999938728, 5728215.0
        ),
    },
    's1': {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999.0,
        'width': SIZE // 16,
        'height': SIZE // 16,
        'count': 3,
        'crs': CRS.from_epsg(32632),
        'transform': Affine(10.0, 0.0, 552245.0, 0.0, -10.0, 5728215.0),
    },
    's2': {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'nodata': None,
        'width': SIZE // 16,
        'height': SIZE // 16,
        'count': 12,
        'crs': CRS.from_epsg(32632),
        'transform': Affine(10.0, 0.0, 552241.6565, 0.0, -10.0, 5728211.6251),
    },
}

multi_labels = {}
for split in ['train', 'test']:
    with open(f'{split}_filenames.lst') as f:
        for filename in f:
            filename = filename.strip()
            for sensor in ['aerial', 's1', 's2']:
                kwargs = profile[sensor]
                directory = os.path.join(sensor, '60m')
                os.makedirs(directory, exist_ok=True)
                if 'int' in kwargs['dtype']:
                    Z = np.random.randint(
                        np.iinfo(kwargs['dtype']).min,
                        np.iinfo(kwargs['dtype']).max,
                        size=(kwargs['height'], kwargs['width']),
                        dtype=kwargs['dtype'],
                    )
                else:
                    Z = np.random.uniform(
                        np.finfo(kwargs['dtype']).min,
                        np.finfo(kwargs['dtype']).max,
                        size=(kwargs['height'], kwargs['width']),
                    )

                path = os.path.join(directory, filename)
                with rasterio.open(path, 'w', **kwargs) as src:
                    for i in range(1, kwargs['count'] + 1):
                        src.write(Z, i)

            k = random.randrange(1, 4)
            labels = random.choices(classes, k=k)
            pcts = np.random.rand(k)
            pcts /= np.sum(pcts)
            multi_labels[filename] = list(map(list, zip(labels, map(float, pcts))))

os.makedirs('labels', exist_ok=True)
path = os.path.join('labels', 'TreeSatBA_v9_60m_multi_labels.json')
with open(path, 'w') as f:
    json.dump(multi_labels, f)

for sensor in ['s1', 's2']:
    shutil.make_archive(sensor, 'zip', '.', sensor)

for spec in species:
    path = f'aerial_60m_{spec}.zip'.lower()
    with zipfile.ZipFile(path, 'w') as f:
        for path in glob.iglob(os.path.join('aerial', '60m', f'{spec}_*.tif')):
            filename = os.path.split(path)[-1]
            f.write(path, arcname=filename)
