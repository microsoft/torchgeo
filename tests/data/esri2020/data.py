#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio

np.random.seed(0)

SIZE = 64


files = [{"image": "N00E020_agb.tif"}, {"image": "N00E020_agb_err.tif"}]


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
    src = rasterio.open(path, "w", **profile)
    src.write(Z)


if __name__ == "__main__":
    dir = "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01"
    tif_name = "00A_20200101-20210101.tif"

    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)

    # Create mask file
    create_file(os.path.join(dir, tif_name), dtype="int8", num_channels=1)

    shutil.make_archive(dir, "zip", base_dir=dir)

    # Compute checksums
    zipfilename = dir + ".zip"
    with open(zipfilename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{zipfilename}: {md5}")
