#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio

SIZE = 32

np.random.seed(0)

dir = "nlcd_2019_land_cover_l48_20210604"


def create_file(path: str, dtype: str):
    """Create the testing file."""
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = 1
    profile["crs"] = "epsg:4326"
    profile["transform"] = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1)
    profile["height"] = SIZE
    profile["width"] = SIZE
    profile["compress"] = "lzw"
    profile["predictor"] = 2

    allowed_values = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]

    Z = np.random.choice(allowed_values, size=(SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)


if __name__ == "__main__":
    # Remove old data
    if os.path.isdir(dir):
        shutil.rmtree(dir)

    os.makedirs(os.path.join(os.getcwd(), dir))

    zip_filename = dir + ".zip"
    filename = dir + ".img"
    create_file(os.path.join(dir, filename), dtype="int8")

    # Compress data
    shutil.make_archive(dir, "zip", ".", dir)

    # Compute checksums
    with open(zip_filename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{zip_filename}: {md5}")

    shutil.rmtree(dir)
