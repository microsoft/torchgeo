#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 128

np.random.seed(0)
files = ["CDL2017_clip.tif", "CDL2018_clip1.tif", "CDL2019_clip.tif"]


def create_file(path: str, dtype: str):
    """Create the testing file."""
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": CRS.from_epsg(32616),
        "transform": Affine(10, 0.0, 399960.0, 0.0, -10, 4500000.0),
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
    dir = os.path.join(os.getcwd())
    os.makedirs(dir, exist_ok=True)

    for file in files:
        create_file(os.path.join(dir, file), dtype="int8")

    # Compute checksums
    for file in files:
        with open(file, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{file}: {md5}")
