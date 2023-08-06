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

filenames: FILENAME_HIERARCHY = {
    "l8biome": {
        "barren": {
            "LC80420082013220LGN00": [
                "LC80420082013220LGN00.TIF",
                "LC80420082013220LGN00_fixedmask.TIF",
            ],
            "LC80530022014156LGN00": [
                "LC80530022014156LGN00.TIF",
                "LC80530022014156LGN00_fixedmask.TIF",
            ],
            "LC81360302014162LGN00": [
                "LC81360302014162LGN00.TIF",
                "LC81360302014162LGN00_fixedmask.TIF",
            ],
        },
        "forest": {
            "LC80070662014234LGN00": [
                "LC80070662014234LGN00.TIF",
                "LC80070662014234LGN00_fixedmask.TIF",
            ],
            "LC80200462014005LGN00": [
                "LC80200462014005LGN00.TIF",
                "LC80200462014005LGN00_fixedmask.TIF",
            ],
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
        "crs": CRS.from_epsg(32615),
        "transform": Affine(30.0, 0.0, 339885.0, 0.0, -30.0, 8286915.0),
    }

    Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])

    if path.endswith("_fixedmask.TIF"):
        Z = np.random.choice(
            np.array([0, 64, 128, 192, 255], dtype=dtype), size=(SIZE, SIZE)
        )
        profile["count"] = 1
    else:
        Z = np.random.randint(256, size=(SIZE, SIZE), dtype=dtype)
        profile["count"] = 11

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

    directories = ["barren", "forest"]
    for directory in directories:
        filename = str(directory)

        # Create tarballs
        shutil.make_archive(filename, "gztar", ".", os.path.join("l8biome", directory))

        # # Compute checksums
        with open(f"{filename}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(filename, md5)
