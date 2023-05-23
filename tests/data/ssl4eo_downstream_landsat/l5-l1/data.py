#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

IMG_DIR = "ssl4eo-l5-l1-conus"
MASK_DIR = "l5-*-2011"
MASKS = ["cdl", "nlcd"]

SUBDIRS = [("0000000", "LT05_045030_20110723"), ("0000001", "LT05_040032_20110805")]

NUM_BANDS = 7
SIZE = 32


def create_image(path: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "width": 264,
        "height": 264,
        "count": 7,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(
            0.00037672803497508636,
            0.0,
            -109.07063613660262,
            0.0,
            -0.0002554026278261721,
            47.49838726154881,
        ),
        "blockysize": 1,
        "tiled": False,
        "compress": "lzw",
        "interleave": "pixel",
    }

    Z = np.random.randint(low=0, high=255, size=(NUM_BANDS, SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z)


def create_mask(path: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "width": 264,
        "height": 264,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(
            0.00037672803497508636,
            0.0,
            -109.07063613660262,
            0.0,
            -0.0002554026278261721,
            47.49838726154881,
        ),
        "blockysize": 1,
        "tiled": False,
        "compress": "lzw",
        "interleave": "band",
    }

    Z = np.random.randint(low=0, high=10, size=(1, SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z)


if __name__ == "__main__":
    # create images
    if os.path.isdir(IMG_DIR):
        shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR, exist_ok=True)

    for subdir in SUBDIRS:
        img_dir = os.path.join(os.getcwd(), IMG_DIR, subdir[0], subdir[1])
        os.makedirs(img_dir)
        create_image(os.path.join(img_dir, "all_bands.tif"))

    shutil.make_archive(IMG_DIR, "gztar", ".", IMG_DIR)

    with open(f"{IMG_DIR}.tar.gz", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(IMG_DIR, md5)

    # create masks
    for mask_name in MASKS:
        mask_dir = MASK_DIR.replace("*", mask_name)
        if os.path.isdir(mask_dir):
            shutil.rmtree(mask_dir)
        os.makedirs(mask_dir, exist_ok=True)

        for subdir in SUBDIRS:
            mask_path = os.path.join(os.getcwd(), mask_dir, subdir[0], subdir[1])
            os.makedirs(mask_path)
            create_mask(os.path.join(mask_path, f"{mask_name}_2011.tif"))

        shutil.make_archive(mask_dir, "gztar", ".", mask_dir)

        with open(f"{mask_dir}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(mask_dir, md5)
