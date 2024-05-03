#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 64  # image width/height

np.random.seed(0)

meta = {
    'driver': 'GTiff',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'crs': CRS.from_epsg(32737),
    'transform': Affine(10.0, 0.0, 390772.3389928384, 0.0, -10.0, 8114182.836060452),
}
count = {'lc': 4, 's1': 2, 's2': 13}
dtype = {'lc': np.uint16, 's1': np.float32, 's2': np.uint16}
stop = {'lc': 11, 's1': np.iinfo(np.uint16).max, 's2': np.iinfo(np.uint16).max}

file_list = []
seasons = ['ROIs1158_spring', 'ROIs1868_summer', 'ROIs1970_fall', 'ROIs2017_winter']
for season in seasons:
    # Remove old data
    if os.path.exists(season):
        shutil.rmtree(season)

    for source in ['lc', 's1', 's2']:
        tarball = f'{season}_{source}.tar.gz'

        # Remove old data
        if os.path.exists(tarball):
            os.remove(tarball)

        directory = os.path.join(season, f'{source}_1')
        os.makedirs(directory)

        # Random images
        for i in range(1, 3):
            filename = f'{season}_{source}_1_p{i}.tif'
            meta['count'] = count[source]
            meta['dtype'] = dtype[source]
            with rasterio.open(os.path.join(directory, filename), 'w', **meta) as f:
                for j in range(1, count[source] + 1):
                    data = np.random.randint(stop[source], size=(SIZE, SIZE)).astype(
                        dtype[source]
                    )
                    f.write(data, j)

            if source == 's2':
                file_list.append(filename)

        # Create tarballs
        shutil.make_archive(f'{season}_{source}', 'gztar', '.', directory)

        # Compute checksums
        with open(tarball, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(repr(md5) + ',')

for split in ['train', 'test']:
    filename = f'{split}_list.txt'

    # Create file list
    with open(filename, 'w') as f:
        for fname in file_list:
            f.write(f'{fname}\n')

    # Compute checksums
    with open(filename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(repr(md5) + ',')
