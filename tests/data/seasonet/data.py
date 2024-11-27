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

np.random.seed(0)

meta = {
    'driver': 'GTiff',
    'nodata': None,
    'crs': CRS.from_epsg(32632),
    'transform': Affine(10.0, 0.0, 664800.0, 0.0, -10.0, 5342400.0),
    'compress': 'zstd',
}
bands = ['10m_RGB', '10m_IR', '20m', '60m', 'labels']
count = {'10m_RGB': 3, '10m_IR': 1, '20m': 6, '60m': 2, 'labels': 1}
dtype = {
    '10m_RGB': np.uint16,
    '10m_IR': np.uint16,
    '20m': np.uint16,
    '60m': np.uint16,
    'labels': np.uint8,
}
size = {'10m_RGB': 120, '10m_IR': 120, '20m': 60, '60m': 20, 'labels': 120}
start = {'10m_RGB': 0, '10m_IR': 0, '20m': 0, '60m': 0, 'labels': 1}
stop = {
    '10m_RGB': np.iinfo(np.uint16).max,
    '10m_IR': np.iinfo(np.uint16).max,
    '20m': np.iinfo(np.uint16).max,
    '60m': np.iinfo(np.uint16).max,
    'labels': 34,
}

meta_lines = [
    'Index,Season,Grid,Latitude,Longitude,Satellite,Year,Month,Day,'
    'Hour,Minute,Second,Clouds,Snow,Classes,SLRAUM,RTYP3,KTYP4,Path\n'
]
seasons = ['spring', 'summer', 'fall', 'winter', 'snow']
grids = [1, 2]
name_comps = [
    ['32UME', '2018', '04', '18', 'T', '10', '40', '21', '53', '928425', '7', '503876'],
    ['32TMT', '2019', '02', '14', 'T', '10', '31', '29', '47', '793488', '7', '808487'],
]
index = 0
for season in seasons:
    # Remove old data
    if os.path.exists(season):
        shutil.rmtree(season)

    archive = f'{season}.zip'

    # Remove old data
    if os.path.exists(archive):
        os.remove(archive)

    for grid, comp in zip(grids, name_comps):
        file_name = f"{comp[0]}_{''.join(comp[1:8])}_{'_'.join(comp[8:])}"
        dir = os.path.join(season, f'grid{grid}', file_name)
        os.makedirs(dir)

        # Random images
        for band in bands:
            meta['count'] = count[band]
            meta['dtype'] = dtype[band]
            meta['width'] = meta['height'] = size[band]
            with rasterio.open(
                os.path.join(dir, f'{file_name}_{band}.tif'), 'w', **meta
            ) as f:
                for j in range(1, count[band] + 1):
                    data = np.random.randint(
                        start[band], stop[band], size=(size[band], size[band])
                    ).astype(dtype[band])
                    f.write(data, j)

        # Generate meta.csv lines
        meta_entries = [
            index,
            season.capitalize(),
            grid,
            f'{comp[8]}.{comp[9]}',
            f'{comp[10]}.{comp[11]}',
            'A',
            comp[1],
            comp[2],
            comp[3],
            comp[5],
            comp[6],
            comp[7],
            0.0,
            0.0,
            "'2,3,12,15,17'",
            1,
            1,
            1,
            dir,
        ]
        meta_lines.append(','.join(map(str, meta_entries)) + '\n')
        index += 1

    # Create archives
    shutil.make_archive(season, 'zip', '.', season)

    # Compute checksums
    with open(archive, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{season}: {md5!r}')

# Write meta.csv
with open('meta.csv', 'w') as f:
    f.writelines(meta_lines)

# Compute checksums
with open('meta.csv', 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(f'meta.csv: {md5!r}')

os.makedirs('splits', exist_ok=True)

for split in ['train', 'val', 'test']:
    filename = f'{split}.csv'

    # Create file list
    with open(os.path.join('splits', filename), 'w') as f:
        for i in range(index):
            f.write(str(i) + '\n')

shutil.make_archive('splits', 'zip', '.', 'splits')

# Compute checksums
with open('splits.zip', 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(f'splits: {md5!r}')
