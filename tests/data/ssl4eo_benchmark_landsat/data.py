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

SIZE = 264

np.random.seed(0)

FILENAME_HIERARCHY = dict[str, "FILENAME_HIERARCHY"] | list[str]

filenames: FILENAME_HIERARCHY = {
    "tm_toa": {
        "0000001": {"LT05_172030_20010526": ["all_bands.tif"]},
        "0000002": {"LT05_223084_20010413": ["all_bands.tif"]},
        "0000003": {"LT05_172034_20020902": ["all_bands.tif"]},
        "0000004": {"LT05_172034_20020903": ["all_bands.tif"]},
        "0000005": {"LT05_172034_20020904": ["all_bands.tif"]},
        "0000006": {"LT05_172034_20020905": ["all_bands.tif"]},
        "0000007": {"LT05_172034_20020906": ["all_bands.tif"]},
        "0000008": {"LT05_172034_20020907": ["all_bands.tif"]},
        "0000009": {"LT05_172034_20020908": ["all_bands.tif"]},
        "0000010": {"LT05_172034_20020909": ["all_bands.tif"]},
    },
    "etm_sr": {
        "0000001": {"LE07_172030_20010526": ["all_bands.tif"]},
        "0000002": {"LE07_223084_20010413": ["all_bands.tif"]},
        "0000003": {"LE07_172034_20020902": ["all_bands.tif"]},
        "0000004": {"LE07_172034_20020903": ["all_bands.tif"]},
        "0000005": {"LE07_172034_20020904": ["all_bands.tif"]},
        "0000006": {"LE07_172034_20020905": ["all_bands.tif"]},
        "0000007": {"LE07_172034_20020906": ["all_bands.tif"]},
        "0000008": {"LE07_172034_20020907": ["all_bands.tif"]},
        "0000009": {"LE07_172034_20020908": ["all_bands.tif"]},
        "0000010": {"LE07_172034_20020909": ["all_bands.tif"]},
    },
    "etm_toa": {
        "0000001": {"LE07_172030_20010526": ["all_bands.tif"]},
        "0000002": {"LE07_223084_20010413": ["all_bands.tif"]},
        "0000003": {"LE07_172034_20020902": ["all_bands.tif"]},
        "0000004": {"LE07_172034_20020903": ["all_bands.tif"]},
        "0000005": {"LE07_172034_20020904": ["all_bands.tif"]},
        "0000006": {"LE07_172034_20020905": ["all_bands.tif"]},
        "0000007": {"LE07_172034_20020906": ["all_bands.tif"]},
        "0000008": {"LE07_172034_20020907": ["all_bands.tif"]},
        "0000009": {"LE07_172034_20020908": ["all_bands.tif"]},
        "0000010": {"LE07_172034_20020909": ["all_bands.tif"]},
    },
    "oli_tirs_toa": {
        "0000001": {"LC08_172030_20010526": ["all_bands.tif"]},
        "0000002": {"LC08_223084_20010413": ["all_bands.tif"]},
        "0000003": {"LC08_172034_20020902": ["all_bands.tif"]},
        "0000004": {"LC08_172034_20020903": ["all_bands.tif"]},
        "0000005": {"LC08_172034_20020904": ["all_bands.tif"]},
        "0000006": {"LC08_172034_20020905": ["all_bands.tif"]},
        "0000007": {"LC08_172034_20020906": ["all_bands.tif"]},
        "0000008": {"LC08_172034_20020907": ["all_bands.tif"]},
        "0000009": {"LC08_172034_20020908": ["all_bands.tif"]},
        "0000010": {"LC08_172034_20020909": ["all_bands.tif"]},
    },
    "oli_sr": {
        "0000001": {"LC08_172030_20010526": ["all_bands.tif"]},
        "0000002": {"LC08_223084_20010413": ["all_bands.tif"]},
        "0000003": {"LC08_172034_20020902": ["all_bands.tif"]},
        "0000004": {"LC08_172034_20020903": ["all_bands.tif"]},
        "0000005": {"LC08_172034_20020904": ["all_bands.tif"]},
        "0000006": {"LC08_172034_20020905": ["all_bands.tif"]},
        "0000007": {"LC08_172034_20020906": ["all_bands.tif"]},
        "0000008": {"LC08_172034_20020907": ["all_bands.tif"]},
        "0000009": {"LC08_172034_20020908": ["all_bands.tif"]},
        "0000010": {"LC08_172034_20020909": ["all_bands.tif"]},
    },
}

num_bands = {"tm_toa": 7, "etm_sr": 6, "etm_toa": 9, "oli_tirs_toa": 11, "oli_sr": 7}
years = {"tm": 2011, "etm": 2019, "oli": 2019}


def create_image(path: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "width": SIZE,
        "height": SIZE,
        "count": num_bands["_".join(path.split(os.sep)[1].split("_")[2:][:-1])],
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

    Z = np.random.randint(low=0, high=255, size=(SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        for i in src.indexes:
            src.write(Z, i)


def create_mask(path: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "width": SIZE,
        "height": SIZE,
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


def create_img_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            if any([x in key for x in filenames.keys()]):
                key = f"ssl4eo_l_{key}_benchmark"
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_img_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_image(path)


def create_mask_directory(
    directory: str, hierarchy: FILENAME_HIERARCHY, mask_product: str
) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_mask_directory(path, value, mask_product)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            year = years[path.split(os.sep)[1].split("_")[2]]
            create_mask(path.replace("all_bands", f"{mask_product}_{year}"))


def create_tarballs(directories) -> None:
    for directory in directories:
        # Create tarballs
        shutil.make_archive(directory, "gztar", ".", directory)

        # Compute checksums
        with open(f"{directory}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(directory, md5)


if __name__ == "__main__":
    # image directories
    create_img_directory(".", filenames)
    directories = filenames.keys()
    directories = [f"ssl4eo_l_{key}_benchmark" for key in directories]
    create_tarballs(directories)

    # mask directory cdl
    mask_keep = ["tm_toa", "etm_sr", "oli_sr"]
    mask_filenames = {
        f"ssl4eo_l_{key.split('_')[0]}_cdl": val
        for key, val in filenames.items()
        if key in mask_keep
    }
    create_mask_directory(".", mask_filenames, "cdl")
    directories = mask_filenames.keys()
    create_tarballs(directories)

    # mask directory nlcd
    mask_filenames = {
        f"ssl4eo_l_{key.split('_')[0]}_nlcd": val
        for key, val in filenames.items()
        if key in mask_keep
    }
    create_mask_directory(".", mask_filenames, "nlcd")
    directories = mask_filenames.keys()
    create_tarballs(directories)
