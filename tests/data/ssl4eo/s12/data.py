#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
from typing import Union

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

SIZE = 36

np.random.seed(0)

FILENAME_HIERARCHY = Union[dict[str, "FILENAME_HIERARCHY"], list[str]]

s1 = ["VH.tif", "VV.tif"]
s2c = [
    "B1.tif",
    "B2.tif",
    "B3.tif",
    "B4.tif",
    "B5.tif",
    "B6.tif",
    "B7.tif",
    "B8.tif",
    "B8A.tif",
    "B9.tif",
    "B10.tif",
    "B11.tif",
    "B12.tif",
]
s2a = s2c.copy()
s2a.remove("B10.tif")
filenames: FILENAME_HIERARCHY = {
    "s1": {
        "0000000": {
            "S1A_IW_GRDH_1SDV_20200329T001515_20200329T001540_031883_03AE27_9BAF": s1,
            "S1A_IW_GRDH_1SDV_20201230T001523_20201230T001548_035908_04349D_C91E": s1,
            "S1B_IW_GRDH_1SDV_20200627T001449_20200627T001514_022212_02A27E_2A09": s1,
            "S1B_IW_GRDH_1SDV_20200928T120105_20200928T120130_023575_02CCB0_F035": s1,
        },
        "0000001": {
            "S1B_IW_GRDH_1SDV_20201101T091054_20201101T091119_024069_02DC0F_F189": s1,
            "S1B_IW_GRDH_1SDV_20210205T091050_20210205T091115_025469_0308CB_AA25": s1,
            "S1B_IW_GRDH_1SDV_20210430T091051_20210430T091116_026694_03303D_69B6": s1,
            "S1B_IW_GRDH_1SDV_20210804T091057_20210804T091122_028094_0359FE_6D9D": s1,
        },
    },
    "s2c": {
        "0000000": {
            "20200323T162931_20200323T163750_T15QXA": s2c,
            "20200621T162901_20200621T164746_T15QXA": s2c,
            "20200924T162929_20200924T164434_T15QXA": s2c,
            "20201228T163711_20201228T164519_T15QXA": s2c,
        },
        "0000001": {
            "20201104T135121_20201104T135117_T21KXT": s2c,
            "20210123T135111_20210123T135113_T21KXT": s2c,
            "20210508T135109_20210508T135519_T21KXT": s2c,
            "20210811T135121_20210811T135115_T21KXT": s2c,
        },
    },
    "s2a": {
        "0000000": {
            "20200323T162931_20200323T163750_T15QXA": s2a,
            "20200621T162901_20200621T164746_T15QXA": s2a,
            "20200924T162929_20200924T164434_T15QXA": s2a,
            "20201228T163711_20201228T164519_T15QXA": s2a,
        },
        "0000001": {
            "20201104T135121_20201104T135117_T21KXT": s2a,
            "20210123T135111_20210123T135113_T21KXT": s2a,
            "20210508T135109_20210508T135519_T21KXT": s2a,
            "20210811T135121_20210811T135115_T21KXT": s2a,
        },
    },
}


def create_file(path: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(
            9.221577104649252e-05,
            0.0,
            -91.84569595740037,
            0.0,
            -8.79720059797404e-05,
            18.588102959877993,
        ),
    }

    if path.endswith("VH.tif") or path.endswith("VV.tif"):
        profile["dtype"] = "float32"

    if path.endswith("B1.tif") or path.endswith("B9.tif") or path.endswith("B10.tif"):
        profile["width"] = profile["height"] = SIZE // 6
        profile["transform"] = Affine(
            0.0005532946262789551,
            0.0,
            -91.84592649682799,
            0.0,
            -0.0005278320358784425,
            18.588322889892943,
        )
    elif (
        path.endswith("B5.tif")
        or path.endswith("B6.tif")
        or path.endswith("B7.tif")
        or path.endswith("B8A.tif")
        or path.endswith("B11.tif")
        or path.endswith("B12.tif")
    ):
        profile["width"] = profile["height"] = SIZE // 2
        profile["transform"] = Affine(
            0.00018443154209298504,
            0.0,
            -91.84574206528589,
            0.0,
            -0.0001759440119594808,
            18.588146945880982,
        )

    Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)


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
            create_file(path)


if __name__ == "__main__":
    create_directory(".", filenames)

    files = ["s1", "s2_l1c", "s2_l2a"]
    directories = ["s1", "s2c", "s2a"]
    for file, directory in zip(files, directories):
        # Create tarballs
        shutil.make_archive(file, "gztar", ".", directory)

        # Compute checksums
        with open(f"{file}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(file, md5)
