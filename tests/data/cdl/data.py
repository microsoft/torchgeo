#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import random
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine

SIZE = 128

np.random.seed(0)
random.seed(0)


def create_file(
    path: str,
    dtype: str,
    num_channels: int,
    *,
    size: int = SIZE,
    crs: str = 'epsg:32616',
    transform: Affine | None = None,
) -> None:
    profile = {}
    profile['driver'] = 'GTiff'
    profile['dtype'] = dtype
    profile['count'] = num_channels
    profile['crs'] = crs
    profile['transform'] = (
        transform
        if transform is not None
        else Affine(30, 0.0, 399960.0, 0.0, -30, 4500000.0)
    )
    profile['height'] = size
    profile['width'] = size
    profile['compress'] = 'lzw'
    profile['predictor'] = 2
    cmap = {
        0: (0, 0, 0, 0),
        1: (255, 211, 0, 255),
        2: (255, 38, 38, 255),
        3: (0, 168, 228, 255),
        4: (255, 158, 11, 255),
        5: (38, 112, 0, 255),
        6: (255, 255, 0, 255),
        7: (0, 0, 0, 255),
        8: (0, 0, 0, 255),
    }

    Z = np.random.randint(size=(size, size), low=0, high=8)

    with rasterio.open(path, 'w', **profile) as src:
        for i in range(1, profile['count'] + 1):
            src.write(Z, i)

        src.write_colormap(1, cmap)


def ensure_cdl_data(
    root: str | os.PathLike[str] | None = None,
    *,
    size: int = SIZE,
    crs: str = 'epsg:32616',
    transform: Affine | None = None,
) -> None:
    """Create fake CDL data at the requested root."""
    root_path = Path(root) if root is not None else Path.cwd()
    directories = ['2023_30m_cdls', '2022_30m_cdls']
    for directory in directories:
        dir_path = root_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        create_file(
            str(dir_path / f'{directory}.tif'),
            dtype='int8',
            num_channels=1,
            size=size,
            crs=crs,
            transform=transform,
        )


directories = ['2023_30m_cdls', '2022_30m_cdls']
raster_extensions = ['.tif', '.tif.ovr']


if __name__ == '__main__':
    for directory in directories:
        filename = directory + '.zip'

        if os.path.isdir(directory):
            shutil.rmtree(directory)

        os.makedirs(os.path.join(os.getcwd(), directory))

        for e in raster_extensions:
            create_file(
                os.path.join(dir, filename.replace('.zip', e)),
                dtype='uint8',
                num_channels=1,
            )

        shutil.make_archive(filename.replace('.zip', ''), 'zip', '.', directory)
