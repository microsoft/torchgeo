#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import random
import zipfile

import numpy as np
import rasterio

np.random.seed(0)
random.seed(0)

SIZE = 64


files = {
    "agb": ["N00E020_agb.tif", "N00E020_agb_err.tif"],
    "gsv": ["N00E020_gsv.tif", "N00E020_gsv_err.tif"],
}


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

    for measurement, file_paths in files.items():
        zipfilename = "N00E020_{}.zip".format(measurement)
        files_to_zip = []
        for path in file_paths:
            # remove old data
            if os.path.exists(path):
                os.remove(path)
            # Create mask file
            create_file(path, dtype="int32", num_channels=1)
            files_to_zip.append(path)

        # Compress data
        with zipfile.ZipFile(zipfilename, "w") as zip:
            for file in files_to_zip:
                zip.write(file, arcname=file)

        # Compute checksums
        with open(zipfilename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{zipfilename}: {md5}")
