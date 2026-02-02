#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import geopandas as gpd
import numpy as np
import pandas as pd

SIZE = 2
COMPRESSION = 'zstd'

np.random.seed(0)

x = np.arange(SIZE)
y = np.arange(SIZE)
t = pd.date_range('2018-01-01', '2018-01-04')
ids = np.arange(SIZE * SIZE)

X, Y = np.meshgrid(x, y)
x = X.flatten()
y = Y.flatten()

# v0
EMBED = 768
directory = 'v0'
filename = '01WCN_20190518_20231021_v001.gpq'
embedding = np.random.rand(SIZE * SIZE, EMBED)
data = {'id': ids, 'date': t, 'embeddings': list(embedding)}
geometry = gpd.points_from_xy(x, y).buffer(0.05).envelope
os.makedirs(directory, exist_ok=True)
gdf = gpd.GeoDataFrame(data, geometry=geometry)
gdf.to_parquet(os.path.join(directory, filename), compression=COMPRESSION)

# v1.5
EMBED = 1024
directory = 'v1.5'
filename = 'data_01c1fab1-0004-5b1c-0009-b72e01d3104e_222_0_0.parquet'
embedding = np.random.rand(SIZE * SIZE, EMBED)
data = {'chips_id': ids, 'datetime': t, 'embedding': list(embedding)}
geometry = gpd.points_from_xy(x, y).buffer(0.01).envelope
os.makedirs(directory, exist_ok=True)
gdf = gpd.GeoDataFrame(data, geometry=geometry)
gdf.to_parquet(os.path.join(directory, filename), compression=COMPRESSION)
