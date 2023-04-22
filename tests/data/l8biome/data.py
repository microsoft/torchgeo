#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
from typing import Union

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

FILENAME_HIERARCHY = Union[dict[str, "FILENAME_HIERARCHY"], list[str]]

bands = [
    "B1.TIF",
    "B2.TIF",
    "B3.TIF",
    "B4.TIF",
    "B5.TIF",
    "B6.TIF",
    "B7.TIF",
    "B8.TIF",
    "B9.TIF",
    "B10.TIF",
    "B11.TIF",
]

filenames: FILENAME_HIERARCHY = {
    "barren": {
        "LC80420082013220LGN00": [],
        "LC80530022014156LGN00": [],
        "LC81360302014162LGN00": [],
    },
    "forest": {"LC80070662014234LGN00": [], "LC80200462014005LGN00": []},
}

for land_type, files in filenames.items():
    for prefix in files:
        for band in bands:
            filenames[land_type][prefix].append(f"{prefix}_{band}")

        filenames[land_type][prefix].append(f"{prefix}_fixedmask.img")


def create_file(path: str) -> None:
    dtype = "uint16"
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": CRS.from_epsg(32615),
        "transform": Affine(30.0, 0.0, 339885.0, 0.0, -30.0, 8286915.0),
    }

    if path.endswith("B8.TIF"):
        profile["transform"] = Affine(15.0, 0.0, 339892.5, 0.0, -15.0, 8286907.5)
        profile["width"] = profile["height"] = SIZE * 2

    Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])

    if path.endswith("fixedmask.img"):
        Z = np.random.randint(5, size=(SIZE, SIZE), dtype=dtype)

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)


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

    directories = ["barren", "forest"]
    for directory in directories:
        filename = str(directory)

        # Create tarballs
        shutil.make_archive(filename, "gztar", ".", directory)

        # # Compute checksums
        with open(f"{filename}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(filename, md5)
