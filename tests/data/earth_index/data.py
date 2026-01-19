#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import geopandas as gpd
import numpy as np

SIZE = 2
EMBED = 384
COMPRESSION = 'zstd'

np.random.seed(0)

x = np.arange(SIZE)
y = np.arange(SIZE)
id_ = np.arange(SIZE * SIZE)
embedding = np.random.rand(SIZE * SIZE, EMBED)

X, Y = np.meshgrid(x, y)
x = X.flatten()
y = Y.flatten()
data = {'id': id_, 'embedding': list(embedding)}

gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(x, y))

directory = '2024'
filename = '01GDM_2024-01-01_2025-01-01.parquet'
os.makedirs(directory, exist_ok=True)
gdf.to_parquet(os.path.join(directory, filename), compression=COMPRESSION)
