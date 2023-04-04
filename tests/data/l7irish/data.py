#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import random
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio import Affine

SIZE = 36

np.random.seed(0)
random.seed(0)

dir = "austral"
patch = "p226_r98"
files = ["L71226098_09820011112_B10.TIF", "L71226098_09820011112_B20.TIF", 
         "L71226098_09820011112_B30.TIF", "L71226098_09820011112_B40.TIF",
         "L71226098_09820011112_B50.TIF", "L71226098_09820011112_B61.TIF", 
         "L72226098_09820011112_B62.TIF", "L72226098_09820011112_B70.TIF",
         "L72226098_09820011112_B80.TIF", "p226_r98_mask2019.TIF"]

def create_file(path: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": CRS.from_epsg(32719),
        "transform": Affine(30.0, 0.0, 462884.99999999994, 0.0, -30.0, 4071915.0)
    }
    
    if path.endswith("B8.TIF"):
        profile["width"] = profile["height"] = SIZE * 2 # resolution = 15m
        profile["transform"]: Affine(15.0, 0.0, 462892.49999999994, 0.0, -15.0, 4071907.5)
    
    if (path.endswith("B61.TIF") or path.endswith("B62.TIF")): 
        profile["width"] = profile["height"] = SIZE // 2  # resolution = 60m
        profile["transform"]: Affine(60.0, 0.0, 462892.49999999994, 0.0, -60.0, 4071907.5)

    Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])

    if path.endswith("_mask2019.TIF"):
         Z = np.random.randint(5, size=(SIZE, SIZE), dtype="uint8")
   
    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)


if __name__ == "__main__":
    # Create images
    filename = dir + ".tar.gz"
    fp = os.path.join(os.getcwd(), "tests", "data", "l7irish", dir)

    # Remove old data
    if os.path.isdir(fp):
        shutil.rmtree(fp)
    
    os.makedirs(fp)
    os.makedirs(os.path.join(fp, patch))

    for file in files:
        create_file(os.path.join(fp, patch, file))
    
    # Compress data
    shutil.make_archive(fp, format="gztar", root_dir=os.path.split(fp)[0], base_dir=dir)

    # Compute checksums
    with open(os.path.join(os.path.split(fp)[0], filename), "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{filename}: {md5}")