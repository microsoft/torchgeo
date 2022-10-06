#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Dict, List, Union

import numpy as np
import rasterio
from rasterio import Affine

SIZE = 36

np.random.seed(0)

FILENAME_HIERARCHY = Union[Dict[str, "FILENAME_HIERARCHY"], List[str]]

filenames: FILENAME_HIERARCHY = {
    # Copernicus Open Access Hub
    "S1A_IW_SLC__1SDV_20221006T133300_20221006T133327_045322_056B2B_84A7.SAFE": {
        "measurement": [
            "s1a-iw1-slc-vh-20221006t133301-20221006t133327-045322-056b2b-001.tiff",
            "s1a-iw1-slc-vv-20221006t133301-20221006t133327-045322-056b2b-004.tiff",
            "s1a-iw2-slc-vh-20221006t133300-20221006t133325-045322-056b2b-002.tiff",
            "s1a-iw2-slc-vv-20221006t133300-20221006t133325-045322-056b2b-005.tiff",
            "s1a-iw3-slc-vh-20221006t133301-20221006t133326-045322-056b2b-003.tiff",
            "s1a-iw3-slc-vv-20221006t133301-20221006t133326-045322-056b2b-006.tiff",
        ]
    }
}


def create_file(path: str) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = "complex64"
    profile["count"] = 1
    profile["crs"] = None
    profile["transform"] = Affine.identity()
    profile["height"] = SIZE
    profile["width"] = SIZE

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
