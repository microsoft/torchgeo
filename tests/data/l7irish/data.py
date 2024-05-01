#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

FILENAME_HIERARCHY = dict[str, "FILENAME_HIERARCHY"] | list[str]

filenames: FILENAME_HIERARCHY = {
    "l7irish": {
        "austral": {
            "p226_r98": ["L71226098_09820011112.TIF", "L7_p226_r98_newmask2015.TIF"],
            "p227_r98": ["L71227098_09820011103.TIF", "L7_p227_r98_newmask2015.TIF"],
            "p231_r93_2": ["L71231093_09320010507.TIF", "L7_p231_r93_newmask2015.TIF"],
        },
        "boreal": {
            "p2_r27": ["L71002027_02720010604.TIF", "L7_p2_r27_newmask2015.TIF"],
            "p143_r21_3": ["L71143021_02120010803.TIF", "L7_p143_r21_newmask2015.TIF"],
        },
    }
}


def create_file(path: str) -> None:
    dtype = "uint8"
    profile = {
        "driver": "COG",
        "compression": "LZW",
        "predictor": 2,
        "dtype": dtype,
        "width": SIZE,
        "height": SIZE,
        "crs": CRS.from_epsg(32719),
        "transform": Affine(30.0, 0.0, 462884.99999999994, 0.0, -30.0, 4071915.0),
    }

    if path.endswith("_newmask2015.TIF"):
        Z = np.random.choice(
            np.array([0, 64, 128, 192, 255], dtype=dtype), size=(SIZE, SIZE)
        )
        profile["count"] = 1
    else:
        Z = np.random.randint(256, size=(SIZE, SIZE), dtype=dtype)
        profile["count"] = 9

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

    directories = ["austral", "boreal"]
    for directory in directories:
        filename = str(directory)

        # Create tarballs
        shutil.make_archive(filename, "gztar", ".", os.path.join("l7irish", directory))

        # # Compute checksums
        with open(f"{filename}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(filename, md5)
