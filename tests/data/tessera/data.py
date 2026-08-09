#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)


def ensure_tessera_data(
    root: str | os.PathLike[str] | None = None,
    *,
    size: int = SIZE,
    crs: CRS | str | None = None,
    transform: Affine | None = None,
    year: str = '2024',
    subfolder: str = 'global_0.1_degree_representation',
) -> None:
    """Create fake Tessera embeddings data using the fixed fixture layout."""
    crs = crs if crs is not None else CRS.from_epsg(32631)
    root_path = Path(root) if root is not None else Path.cwd()
    directory = root_path / subfolder / year / 'grid_0.05_51.35'
    filename = directory / f'grid_0.05_51.35_{year}.tiff'

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': size,
        'height': size,
        'count': 128,
        'crs': crs,
        'transform': transform
        if transform is not None
        else Affine(10.0, 0.0, 290872.40803907975, 0.0, -10.0, 5698579.144861946),
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True,
        'compress': 'lzw',
        'interleave': 'pixel',
    }

    directory.mkdir(parents=True, exist_ok=True)
    Z = np.random.random(size=(profile['count'], size, size)) * 2 - 1
    with rasterio.open(filename, 'w', **profile) as src:
        src.write(Z)


if __name__ == '__main__':
    ensure_tessera_data()
