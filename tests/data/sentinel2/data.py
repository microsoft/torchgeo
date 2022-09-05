#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Dict, List, Union

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

FILENAME_HIERARCHY = Union[Dict[str, "FILENAME_HIERARCHY"], List[str]]

filenames: FILENAME_HIERARCHY = {
    # USGS Earth Explorer
    "S2A_MSIL1C_20220412T162841_N0400_R083_T16TFM_20220412T202300.SAFE": {
        "GRANULE": {
            "L1C_T16TFM_A035544_20220412T163959": {
                "IMG_DATA": [
                    "T16TFM_20220412T162841_B01.jp2",
                    "T16TFM_20220412T162841_B02.jp2",
                    "T16TFM_20220412T162841_B03.jp2",
                    "T16TFM_20220412T162841_B04.jp2",
                    "T16TFM_20220412T162841_B05.jp2",
                    "T16TFM_20220412T162841_B06.jp2",
                    "T16TFM_20220412T162841_B07.jp2",
                    "T16TFM_20220412T162841_B08.jp2",
                    "T16TFM_20220412T162841_B09.jp2",
                    "T16TFM_20220412T162841_B10.jp2",
                    "T16TFM_20220412T162841_B11.jp2",
                    "T16TFM_20220412T162841_B12.jp2",
                    "T16TFM_20220412T162841_B8A.jp2",
                    "T16TFM_20220412T162841_TCI.jp2",
                ]
            }
        }
    },
    # Copernicus Open Access Hub
    "S2A_MSIL2A_20220414T110751_N0400_R108_T26EMU_20220414T165533.SAFE": {
        "GRANULE": {
            "L2A_T26EMU_A035569_20220414T110747": {
                "IMG_DATA": {
                    "R10m": [
                        "T26EMU_20220414T110751_AOT_10m.jp2",
                        "T26EMU_20220414T110751_B02_10m.jp2",
                        "T26EMU_20220414T110751_B03_10m.jp2",
                        "T26EMU_20220414T110751_B04_10m.jp2",
                        "T26EMU_20220414T110751_B08_10m.jp2",
                        "T26EMU_20220414T110751_TCI_10m.jp2",
                        "T26EMU_20220414T110751_WVP_10m.jp2",
                    ],
                    "R20m": [
                        "T26EMU_20220414T110751_AOT_20m.jp2",
                        "T26EMU_20220414T110751_B01_20m.jp2",
                        "T26EMU_20220414T110751_B02_20m.jp2",
                        "T26EMU_20220414T110751_B03_20m.jp2",
                        "T26EMU_20220414T110751_B04_20m.jp2",
                        "T26EMU_20220414T110751_B05_20m.jp2",
                        "T26EMU_20220414T110751_B06_20m.jp2",
                        "T26EMU_20220414T110751_B07_20m.jp2",
                        "T26EMU_20220414T110751_B11_20m.jp2",
                        "T26EMU_20220414T110751_B12_20m.jp2",
                        "T26EMU_20220414T110751_B8A_20m.jp2",
                        "T26EMU_20220414T110751_SCL_20m.jp2",
                        "T26EMU_20220414T110751_TCI_20m.jp2",
                        "T26EMU_20220414T110751_WVP_20m.jp2",
                    ],
                    "R60m": [
                        "T26EMU_20220414T110751_AOT_60m.jp2",
                        "T26EMU_20220414T110751_B01_60m.jp2",
                        "T26EMU_20220414T110751_B02_60m.jp2",
                        "T26EMU_20220414T110751_B03_60m.jp2",
                        "T26EMU_20220414T110751_B04_60m.jp2",
                        "T26EMU_20220414T110751_B05_60m.jp2",
                        "T26EMU_20220414T110751_B06_60m.jp2",
                        "T26EMU_20220414T110751_B07_60m.jp2",
                        "T26EMU_20220414T110751_B09_60m.jp2",
                        "T26EMU_20220414T110751_B11_60m.jp2",
                        "T26EMU_20220414T110751_B12_60m.jp2",
                        "T26EMU_20220414T110751_B8A_60m.jp2",
                        "T26EMU_20220414T110751_SCL_60m.jp2",
                        "T26EMU_20220414T110751_TCI_60m.jp2",
                        "T26EMU_20220414T110751_WVP_60m.jp2",
                    ],
                }
            }
        }
    },
}


def create_file(path: str, dtype: str, num_channels: int) -> None:
    res = 10
    root, _ = os.path.splitext(path)
    if root.endswith("m"):
        res = int(root[-3:-1])

    profile = {}
    profile["driver"] = "JP2OpenJPEG"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = CRS.from_epsg(32616)
    profile["transform"] = Affine(res, 0.0, 399960.0, 0.0, -res, 4500000.0)
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
            create_file(path, dtype="uint16", num_channels=1)


if __name__ == "__main__":
    create_directory(".", filenames)
