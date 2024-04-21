#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 128
current_x_offset = 0

np.random.seed(0)

FILENAME_HIERARCHY = dict[str, dict | list[str]]

filenames: FILENAME_HIERARCHY = {}

base_path = "S2A_MSIL1C_20220412T162841_N0400_R083_{tile_id}_20220412T202300.SAFE"
granule_path = "L1C_{tile_id}_A035544_20220412T163959"
bands = [
    "B01.jp2",
    "B02.jp2",
    "B03.jp2",
    "B04.jp2",
    "B05.jp2",
    "B06.jp2",
    "B07.jp2",
    "B08.jp2",
    "B09.jp2",
    "B10.jp2",
    "B11.jp2",
    "B12.jp2",
    "B8A.jp2",
    "TCI.jp2",
]

for i in range(16, 26):
    tile_id = f"T{i}TFM"
    safe_file = base_path.format(tile_id=tile_id)
    granule = granule_path.format(tile_id=tile_id)
    img_data = [f"{tile_id}_20220412T162841_{band}" for band in bands] + [
        f"{tile_id}_20190412T162841_{band}" for band in bands
    ]

    filenames[safe_file] = {"GRANULE": {granule: {"IMG_DATA": img_data}}}


def create_file(path: str, dtype: str, num_channels: int, x_offset: int) -> None:
    res = 10
    root, _ = os.path.splitext(path)
    if root.endswith("m"):
        res = int(root[-3:-1])

    profile = {}
    profile["driver"] = "JP2OpenJPEG"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = CRS.from_epsg(32616)
    profile["transform"] = Affine(res, 0.0, 399960.0 + x_offset, 0.0, -res, 4500000.0)
    profile["height"] = round(SIZE * 10 / res)
    profile["width"] = round(SIZE * 10 / res)

    if "float" in profile["dtype"]:
        Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])
    else:
        Z = np.random.randint(
            np.iinfo(profile["dtype"]).max, size=(SIZE, SIZE), dtype=profile["dtype"]
        )

    with rasterio.open(path, "w", **profile) as src:
        for i in range(1, profile["count"] + 1):
            src.write(Z, i)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    global current_x_offset

    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        print(current_x_offset)
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path, dtype="uint16", num_channels=1, x_offset=current_x_offset)
        current_x_offset += SIZE


if __name__ == "__main__":
    create_directory(".", filenames)
