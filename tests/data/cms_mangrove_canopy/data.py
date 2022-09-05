#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import random
import shutil

import numpy as np
import rasterio

np.random.seed(0)
random.seed(0)

SIZE = 64


files = [
    {"image": "Mangrove_agb_Angola.tif"},
    {"image": "Mangrove_hba95_Angola.tif"},
    {"image": "Mangrove_hmax95_Angola.tif"},
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

    Z = np.random.randint(
        np.iinfo(profile["dtype"]).max, size=(1, SIZE, SIZE), dtype=profile["dtype"]
    )
    with rasterio.open(path, "w", **profile) as src:
        src.write(Z)


if __name__ == "__main__":
    directory = "CMS_Global_Map_Mangrove_Canopy_1665"

    # Remove old data
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(os.path.join(directory, "data"), exist_ok=True)

    for file_dict in files:
        # Create mask file
        path = file_dict["image"]
        create_file(
            os.path.join(directory, "data", path), dtype="int32", num_channels=1
        )

    # Compress data
    shutil.make_archive(directory.replace(".zip", ""), "zip", ".", directory)

    # Compute checksums
    with open(directory + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{directory}: {md5}")
