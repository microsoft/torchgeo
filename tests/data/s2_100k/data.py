#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 32
COUNT = 1

rng = np.random.default_rng(0)

profile = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'nodata': None,
    'width': SIZE,
    'height': SIZE,
    'count': 12,
    'crs': CRS.from_epsg(32639),
    'transform': Affine(10.0, 0.0, 524385.0, 0.0, -10.0, 2712815.0),
    'blockxsize': 512,
    'blockysize': 512,
    'tiled': True,
    'compress': 'lzw',
    'interleave': 'pixel',
}

Z = rng.integers(
    np.iinfo(profile['dtype']).max,
    size=(profile['count'], profile['height'], profile['width']),
    dtype=profile['dtype'],
)

images = Path('images')
images.mkdir(exist_ok=True)
fn = []
for i in range(COUNT):
    patch = f'patch_{i}.tif'
    fn.append(patch)
    with rio.open(images / patch, 'w', **profile) as src:
        src.write(Z)

shutil.make_archive('satclip', 'tar', images)

lon = rng.uniform(-180, 180, COUNT)
lat = rng.uniform(-90, 90, COUNT)

df = pd.DataFrame({'fn': fn, 'lon': lon, 'lat': lat})
df.to_csv('index.csv', index=False)
