# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
import rasterio.transform

SIZE = 32


def generate_test_data(fn: str) -> str:
    """Creates test data with uint32 datatype.

    Args:
        fn (str): Filename to write

    Returns:
        str: md5 hash of created archive
    """
    profile = {
        "driver": "GTiff",
        "dtype": "uint32",
        "count": 1,
        "crs": "epsg:4326",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1),
        "height": SIZE,
        "width": SIZE,
        "compress": "lzw",
        "predictor": 2,
    }

    with rasterio.open(fn, "w", **profile) as f:
        f.write(np.random.randint(0, 64, size=(1, SIZE, SIZE)))


if __name__ == "__main__":
    directory = "ts_data"
    # Remove old data
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    dates = ["20220101", "20220102", "20220103", "20220104", "20220105"]
    bands = ["B04", "B03", "B02", "target"]
    for d in dates:
        for b in bands:
            fn = os.path.join(directory, f"test_{d}_{b}.tif")
            generate_test_data(fn)

    # Compress data
    shutil.make_archive(directory.replace(".zip", ""), "zip", ".", directory)

    # Compute checksums
    with open(directory + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{directory}: {md5}")
