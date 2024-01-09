#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
from typing import Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32

np.random.seed(0)
num_bands = 13

FILENAME_HIERARCHY = Union[dict[str, "FILENAME_HIERARCHY"], list[str]]

filenames: FILENAME_HIERARCHY = {
    "imgs": {
        "0000002": {
            "20180329T030539_20180329T030537_T50UPU": ["all_bands.tif"],
            "20191025T030811_20191025T030910_T50UPU": ["all_bands.tif"],
            "20180912T025539_20180912T030310_T50UPU": ["all_bands.tif"],
        },
        "0008639": {
            "20191022T025751_20191022T030336_T51UVQ": ["all_bands.tif"],
            "20180719T025551_20180719T025950_T51UVQ": ["all_bands.tif"],
        },
    }
}


def create_file(path: str):
    """Create the testing file."""
    dtype = "uint8"
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(0.00011742899566642832, 0.0, 122.06360523941697,
                            0.0, -8.896045739291803e-05, 41.183714914402124),
        "height": SIZE,
        "width": SIZE,
        "compress": "lzw",
        "predictor": 2,
    }

    Z = np.random.choice(4095, size=(SIZE, SIZE))
    profile["count"] = num_bands

    with rasterio.open(path, "w", **profile) as src:
        for i in range(1, profile["count"] + 1):
            src.write(Z, i)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path)


if __name__ == "__main__":
    create_directory(".", filenames)

    # Create tarballs
    shutil.make_archive("imgs", "gztar", ".", "imgs")

    # Compute checksums
    with open("imgs.tar.gz", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print("imgs.tar.gz", md5)