#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import random
import shutil

import numpy as np
import rasterio

SIZE = 32

np.random.seed(0)
random.seed(0)


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
    cmap = {
        0: (0, 0, 0, 0),
        1: (255, 211, 0, 255),
        2: (255, 38, 38, 255),
        3: (0, 168, 228, 255),
        4: (255, 158, 11, 255),
        5: (38, 112, 0, 255),
        6: (255, 255, 0, 255),
        7: (0, 0, 0, 255),
        8: (0, 0, 0, 255),
    }

    Z = np.random.randint(size=(SIZE, SIZE), low=0, high=8)

    src = rasterio.open(path, "w", **profile)
    for i in range(1, profile["count"] + 1):
        src.write(Z, i)

    src.write_colormap(1, cmap)


directories = ["2020_30m_cdls", "2021_30m_cdls"]
raster_extensions = [".tif", ".tif.ovr"]


if __name__ == "__main__":

    for dir in directories:
        filename = dir + ".zip"

        # Remove old data
        if os.path.isdir(dir):
            shutil.rmtree(dir)

        os.makedirs(os.path.join(os.getcwd(), dir))

        for e in raster_extensions:
            create_file(
                os.path.join(dir, filename.replace(".zip", e)),
                dtype="int8",
                num_channels=1,
            )

        # Compress data
        shutil.make_archive(filename.replace(".zip", ""), "zip", ".", dir)

        # Compute checksums
        with open(filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{filename}: {md5}")

        shutil.rmtree(dir)
