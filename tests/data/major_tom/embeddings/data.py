#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import geopandas as gpd
import numpy as np
import pandas as pd

SIZE = 2
EMBED = 2048
COMPRESSION = 'snappy'

np.random.seed(0)

x = np.arange(SIZE)
y = np.arange(SIZE)
t = pd.date_range('2018-01-01', '2018-01-04')
embedding = np.random.rand(SIZE * SIZE, EMBED)

X, Y = np.meshgrid(x, y)
x = X.flatten()
y = Y.flatten()
data = {'embedding': list(embedding), 'centre_lon': x, 'centre_lat': y, 'timestamp': t}
geometry = gpd.points_from_xy(x, y).buffer(0.5).envelope

gdf = gpd.GeoDataFrame(data, geometry=geometry)

directory = 'embeddings'
filename = 'part_00001-00050.parquet'
os.makedirs(directory, exist_ok=True)
gdf.to_parquet(os.path.join(directory, filename), compression=COMPRESSION)
