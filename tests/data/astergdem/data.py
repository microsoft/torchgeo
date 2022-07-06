# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import zipfile

import numpy as np
import rasterio

np.random.seed(0)

SIZE = 64

files = [
    {"image": "ASTGTMV003_N000000_dem.tif"},
    {"image": "ASTGTMV003_N000010_dem.tif"},
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
    src = rasterio.open(path, "w", **profile)
    src.write(Z)


if __name__ == "__main__":
    zipfilename = "astergdem.zip"
    files_to_zip = []

    for file_dict in files:
        path = file_dict["image"]
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
