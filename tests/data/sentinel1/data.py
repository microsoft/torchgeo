#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

FILENAME_HIERARCHY = dict[str, "FILENAME_HIERARCHY"] | list[str]

filenames: FILENAME_HIERARCHY = {
    # ASF DAAC
    "S1A_IW_20221204T161641_DVR_RTC30_G_gpuned_1AE1": [
        "S1A_IW_20221204T161641_DVR_RTC30_G_gpuned_1AE1_VH.tif",
        "S1A_IW_20221204T161641_DVR_RTC30_G_gpuned_1AE1_VV.tif",
    ],
    "S1B_IW_20161021T042948_DHP_RTC30_G_gpuned_A784": [
        "S1B_IW_20161021T042948_DHP_RTC30_G_gpuned_A784_HH.tif",
        "S1B_IW_20161021T042948_DHP_RTC30_G_gpuned_A784_HV.tif",
    ],
}


def create_file(path: str) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = "float32"
    profile["count"] = 1
    profile["crs"] = CRS.from_epsg(32605)
    profile["transform"] = Affine(30.0, 0.0, 79860.0, 0.0, -30.0, 2298240.0)
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
