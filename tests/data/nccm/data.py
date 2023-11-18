#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32

np.random.seed(0)
files = ["CDL2017_clip.tif", "CDL2018_clip1.tif", "CDL2019_clip.tif"]


def create_file(path: str, dtype: str):
    """Create the testing file."""
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(
            8.983152841195208e-05,
            0.0,
            115.483402043364,
            0.0,
            -8.983152841195208e-05,
            53.531397320113605,
        ),
        "height": SIZE,
        "width": SIZE,
        "compress": "lzw",
        "predictor": 2,
    }

    allowed_values = [0, 1, 2, 3, 15]

    Z = np.random.choice(allowed_values, size=(SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)


if __name__ == "__main__":
    dir = os.path.join(os.getcwd(), "13090442")

    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)

    os.makedirs(dir, exist_ok=True)

    for file in files:
        create_file(os.path.join(dir, file), dtype="int8")

    # Compress data
    shutil.make_archive("13090442", "zip", ".", dir)

    # Compute checksums
    with open("13090442.zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"13090442.zip: {md5}")
