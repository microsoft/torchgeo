#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import geopandas as gpd
import numpy as np
import pandas as pd

SIZE = 2
EMBED = 768
COMPRESSION = 'zstd'

np.random.seed(0)

x = np.arange(SIZE)
y = np.arange(SIZE)
t = pd.date_range('2018-01-01', '2018-01-04')
ids = np.arange(SIZE * SIZE)
embedding = np.random.rand(SIZE * SIZE, EMBED)

X, Y = np.meshgrid(x, y)
x = X.flatten()
y = Y.flatten()
data = {'id': ids, 'date': t, 'embeddings': list(embedding)}
geometry = gpd.points_from_xy(x, y).buffer(0.05).envelope

gdf = gpd.GeoDataFrame(data, geometry=geometry)

directory = 'data'
filename = '01WCN_20190518_20231021_v001.gpq'
os.makedirs(directory, exist_ok=True)
gdf.to_parquet(os.path.join(directory, filename), compression=COMPRESSION)
