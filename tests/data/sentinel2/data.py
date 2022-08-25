#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio

SIZE = 32

np.random.seed(0)

filenames = [
    "T41XNE_20200829T083611_B01_60m.tif",
    "T41XNE_20200829T083611_B02_10m.tif",
    "T41XNE_20200829T083611_B03_10m.tif",
    "T41XNE_20200829T083611_B04_10m.tif",
    "T41XNE_20200829T083611_B05_20m.tif",
    "T41XNE_20200829T083611_B06_20m.tif",
    "T41XNE_20200829T083611_B07_20m.tif",
    "T41XNE_20200829T083611_B08_10m.tif",
    "T41XNE_20200829T083611_B8A_20m.tif",
    "T41XNE_20200829T083611_B09_60m.tif",
    "T41XNE_20200829T083611_B11_20m.tif",
]


def create_file(path: str, dtype: str, num_channels: int) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = "epsg:4326"
    profile["transform"] = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1)
    profile["height"] = SIZE
    profile["width"] = SIZE
    profile["compress"] = "lzw"
    profile["predictor"] = 2

    if "float" in profile["dtype"]:
        Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])
    else:
        Z = np.random.randint(
            np.iinfo(profile["dtype"]).max, size=(SIZE, SIZE), dtype=profile["dtype"]
        )

    src = rasterio.open(path, "w", **profile)
    for i in range(1, profile["count"] + 1):
        src.write(Z, i)


if __name__ == "__main__":
    for f in filenames:
        if os.path.exists(f):
            os.remove(f)

        create_file(path=f, dtype="int32", num_channels=1)
