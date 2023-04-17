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
    "B10.TIF",
    "B20.TIF",
    "B30.TIF",
    "B40.TIF",
    "B50.TIF",
    "B61.TIF",
    "B62.TIF",
    "B70.TIF",
    "B80.TIF",
]

filenames: FILENAME_HIERARCHY = {
    "austral": {"p226_r98": [], "p227_r98": [], "p231_r93_2": []},
    "boreal": {"p2_r27": [], "p143_r21_3": []},
}
prefixes = [
    "L71226098_09820011112",
    "L71227098_09820011103",
    "L71231093_09320010507",
    "L71002027_02720010604",
    "L71143021_02120010803",
]

for land_type, patches in filenames.items():
    for patch in patches:
        path, row = patch.split("_")[:2]
        key = path[1:].zfill(3) + row[1:].zfill(3)
        for prefix in prefixes:
            if key in prefix:
                for band in bands:
                    if band in ["B62.TIF", "B70.TIF", "B80.TIF"]:
                        prefix = prefix.replace("L71", "L72")
                    filenames[land_type][patch].append(f"{prefix}_{band}")

        filenames[land_type][patch].append(f"L7_{path}_{row}_newmask2015.TIF")


def create_file(path: str) -> None:
    dtype = "uint8"
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": CRS.from_epsg(32719),
        "transform": Affine(30.0, 0.0, 462884.99999999994, 0.0, -30.0, 4071915.0),
    }

    if path.endswith("B80.TIF"):
        profile["transform"] = Affine(
            15.0, 0.0, 462892.49999999994, 0.0, -15.0, 4071907.5
        )
        profile["width"] = profile["height"] = SIZE * 2

    if path.endswith("_newmask2015.TIF"):
        Z = np.random.choice(
            np.array([0, 64, 128, 191, 255], dtype=dtype), size=(SIZE, SIZE)
        )

    else:
        Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])

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

    directories = ["austral", "boreal"]
    for directory in directories:
        filename = str(directory)

        # Create tarballs
        shutil.make_archive(filename, "gztar", ".", directory)

        # # Compute checksums
        with open(f"{filename}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(filename, md5)
