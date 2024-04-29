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

# USGS Earth Explorer
base_path = "S2A_MSIL1C_20220412T162841_N0400_R083_{tile_id}_20220412T202300.SAFE"
granule_path = "L1C_{tile_id}_A035544_20220412T163959"
bands = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
    "TCI",
]

for i in range(16, 26):
    tile_id = f"T{i}TFM"
    safe_file = base_path.format(tile_id=tile_id)
    granule = granule_path.format(tile_id=tile_id)
    img_data = [f"{tile_id}_20220412T162841_{band}.jp2" for band in bands] + [
        f"{tile_id}_20190412T162841_{band}.jp2" for band in bands
    ]

    filenames[safe_file] = {"GRANULE": {granule: {"IMG_DATA": img_data}}}

# Copernicus Open Access Hub
base_path = "S2A_MSIL2A_20220414T110751_N0400_R108_{tile_id}_20220414T165533.SAFE"
granule_path = "L2A_{tile_id}_A035569_20220414T110747"
resolutions = ["10m", "20m", "60m"]
bands_10m = ["AOT", "B02", "B03", "B04", "B08", "TCI", "WVP"]
bands_20m = [
    "AOT",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B11",
    "B12",
    "B8A",
    "SCL",
    "TCI",
    "WVP",
]
bands_60m = [
    "AOT",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B09",
    "B11",
    "B12",
    "B8A",
    "SCL",
    "TCI",
    "WVP",
]

for resolution in resolutions:
    if resolution == "10m":
        bands = bands_10m
    elif resolution == "20m":
        bands = bands_20m
    else:
        bands = bands_60m

    for i in range(26, 36):
        tile_id = f"T{i}EMU"
        safe_file = base_path.format(tile_id=tile_id)
        granule = granule_path.format(tile_id=tile_id)
        img_data = [
            f"{tile_id}_20220414T110751_{band}_{resolution}.jp2" for band in bands
        ] + [f"{tile_id}_20190414T110751_{band}_{resolution}.jp2" for band in bands]

        if safe_file not in filenames:
            filenames[safe_file] = {"GRANULE": {}}
        if granule not in filenames[safe_file]["GRANULE"]:
            filenames[safe_file]["GRANULE"][granule] = {"IMG_DATA": {}}

        filenames[safe_file]["GRANULE"][granule]["IMG_DATA"][
            f"R{resolution}"
        ] = img_data


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
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path, dtype="uint16", num_channels=1, x_offset=current_x_offset)
        current_x_offset += SIZE


if __name__ == "__main__":
    create_directory(".", filenames)
