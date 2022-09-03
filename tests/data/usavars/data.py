#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import rasterio

data_dir = "uar"
labels = [
    "elevation",
    "population",
    "treecover",
    "income",
    "nightlights",
    "housing",
    "roads",
]
splits = ["train", "val", "test"]

SIZE = 3


def create_file(path: str, dtype: str, num_channels: int) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = "epsg:4326"
    profile["transform"] = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1)
    profile["height"] = SIZE
    profile["width"] = SIZE
    profile["compress"] = "lzw"
    profile["predictor"] = 2

    Z = np.random.randint(
        np.iinfo(profile["dtype"]).max, size=(4, SIZE, SIZE), dtype=profile["dtype"]
    )
    with rasterio.open(path, "w", **profile) as src:
        src.write(Z)


# Remove old data
filename = f"{data_dir}.zip"
csvs = glob.glob("*.csv")
txts = glob.glob("*.txt")

for csv in csvs:
    os.remove(csv)
for txt in txts:
    os.remove(txt)
if os.path.exists(filename):
    os.remove(filename)
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

# Create tifs:
os.makedirs(data_dir)
create_file(os.path.join(data_dir, "tile_0,0.tif"), np.uint8, 4)
create_file(os.path.join(data_dir, "tile_0,1.tif"), np.uint8, 4)

# Create labels:
columns = [["ID", "lon", "lat", lab] for lab in labels]
fake_vals = [["0,0", 0.0, 0.0, 0.0], ["0,1", 0.1, 0.1, 1.0]]
for lab, cols in zip(labels, columns):
    df = pd.DataFrame(fake_vals, columns=cols)
    df.to_csv(lab + ".csv")

# Create splits:
with open("train_split.txt", "w") as f:
    f.write("tile_0,0.tif" + "\n")
    f.write("tile_0,0.tif" + "\n")
    f.write("tile_0,0.tif" + "\n")
with open("val_split.txt", "w") as f:
    f.write("tile_0,1.tif" + "\n")
    f.write("tile_0,1.tif" + "\n")
with open("test_split.txt", "w") as f:
    f.write("tile_0,0.tif" + "\n")

# Compress data
shutil.make_archive(data_dir, "zip", ".", data_dir)

# Compute checksums
filename = f"{data_dir}.zip"
with open(filename, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print(repr(filename) + ": " + repr(md5) + ",")
