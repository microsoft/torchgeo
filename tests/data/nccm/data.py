#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32

np.random.seed(0)

dir = "CDL{}_clip"

years = [2017,2018,2019]

def create_file(path: str, dtype: str):
    """Create the testing file."""
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(8.983152841195208e-05, 0.0, 115.483402043364,0.0, -8.983152841195208e-05, 53.531397320113605),
        "height": SIZE,
        "width": SIZE,
        "compress": "lzw",
        "predictor": 2,
    }

    allowed_values = [0,1,2,3]

    Z = np.random.choice(allowed_values, size=(SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)


if __name__ == "__main__":
    for year in years:
        year_dir = dir.format(year)
        # Remove old data
        if os.path.isdir(year_dir):
            shutil.rmtree(year_dir)

        os.makedirs(os.path.join(os.getcwd(), year_dir))

        zip_filename = year_dir + ".zip"
        filename = year_dir + ".tif"
        create_file(os.path.join(year_dir, filename), dtype="int8")

        # Compress data
        shutil.make_archive(year_dir, "zip", ".", year_dir)

        # Compute checksums
        with open(zip_filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{zip_filename}: {md5}")