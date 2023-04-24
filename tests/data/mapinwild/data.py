#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32

np.random.seed(0)

meta = {
    "driver": "GTiff",
    "nodata": None,
    "width": SIZE,
    "height": SIZE,
    "crs": CRS.from_epsg(32720),
    "transform": Affine(10.0, 0.0, 612190.0, 0.0, -10.0, 7324250.0),
}

count = {
    "ESA_WC": 1,
    "VIIRS": 1,
    "mask": 1,
    "s1": 2,
    "s1": 2,
    "s2_temporal_subset": 10,
    "s2_autumn": 10,
    "s2_spring": 10,
    "s2_summer": 10,
    "s2_winter": 10,
}
dtype = {
    "ESA_WC": np.uint8,
    "VIIRS": np.float32,
    "mask": np.byte,
    "s1": np.float64,
    "s2_temporal_subset": np.uint16,
    "s2_autumn": np.uint16,
    "s2_spring": np.uint16,
    "s2_summer": np.uint16,
    "s2_winter": np.uint16,
}
stop = {
    "ESA_WC": np.iinfo(np.uint8).max,
    "VIIRS": np.finfo(np.float32).max,
    "mask": np.iinfo(np.byte).max,
    "s1": np.finfo(np.float64).max,
    "s2_temporal_subset": np.iinfo(np.uint16).max,
    "s2_autumn": np.iinfo(np.uint16).max,
    "s2_spring": np.iinfo(np.uint16).max,
    "s2_summer": np.iinfo(np.uint16).max,
    "s2_winter": np.iinfo(np.uint16).max,
}

folder_path = os.path.join(os.getcwd(), "tests", "data", "mapinwild")


for source in [
    "ESA_WC",
    "VIIRS",
    "mask",
    "s1",
    "s2_temporal_subset",
    "s2_autumn",
    "s2_spring",
    "s2_summer",
    "s2_winter",
]:
    directory = os.path.join(folder_path, source)

    # Remove old data
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    # Random images
    for i in range(1, 3):
        filename = f"{i}.tif"
        filepath = os.path.join(directory, filename)

        meta["count"] = count[source]
        meta["dtype"] = dtype[source]
        with rasterio.open(filepath, "w", **meta) as f:
            for j in range(1, count[source] + 1):
                if meta["dtype"] is np.float32 or meta["dtype"] is np.float64:
                    data = np.random.randn(SIZE, SIZE).astype(dtype[source])

                else:
                    data = np.random.randint(stop[source], size=(SIZE, SIZE)).astype(
                        dtype[source]
                    )
                f.write(data, j)

    # Compress data
    shutil.make_archive(directory.replace(".zip", ""), "zip", ".", directory)

    # Compute checksums
    with open(filepath, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{filepath}: {md5}")

tvt_split = pd.DataFrame(
    [["1", "2", "3"], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    index=["0", "1", "2"],
    columns=["train", "validation", "test"],
)
tvt_split.dropna()
tvt_split.to_csv(os.path.join(folder_path, "split_IDs.csv"))
