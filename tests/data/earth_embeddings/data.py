#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import pandas as pd

SIZE = 2
EMBED = 1024
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

df = pd.DataFrame(data)

directory = 'dinov2'
filename = 'DINOv2_grid_sample_center_384x384_244k.parquet'
os.makedirs(directory, exist_ok=True)
df.to_parquet(os.path.join(directory, filename), compression=COMPRESSION)
