#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def generate_test_data() -> str:
    """Create test data archive for AgriFieldNet dataset.
    Args:
        paths: path to store test data
        n_samples: number of samples.
    Returns:
        md5 hash of created archive
    """
    paths = "south_africa_crop_type"
    dtype = np.uint8
    dtype_max = np.iinfo(dtype).max

    SIZE = 256

    np.random.seed(0)

    s1_bands = ("VH", "VV")
    s2_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    )

    profile = {
        "dtype": dtype,
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": CRS.from_epsg(32634),
        "transform": Affine(10.0, 0.0, 535840.0, 0.0, -10.0, 3079680.0),
    }

    train_imagery_s1_dir = os.path.join(paths, "train", "imagery", "s1")
    train_imagery_s2_dir = os.path.join(paths, "train", "imagery", "s2")
    train_labels_dir = os.path.join(paths, "train", "labels")

    os.makedirs(train_imagery_s1_dir, exist_ok=True)
    os.makedirs(train_imagery_s2_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    train_field_ids = ["12"]

    s1_timestamps = ["2017_04_01", "2017_07_28"]
    s2_timestamps = ["2017_05_04", "2017_07_22"]

    def write_raster(path: str, arr: np.array) -> None:
        with rasterio.open(path, "w", **profile) as src:
            src.write(arr, 1)

    for field_id in train_field_ids:
        for date in s1_timestamps:
            s1_dir = os.path.join(train_imagery_s1_dir, field_id, date)
            os.makedirs(s1_dir, exist_ok=True)
            for band in s1_bands:
                train_arr = np.random.randint(
                    dtype_max, size=(SIZE, SIZE), dtype=dtype
                )  # noqa: E501
                path = os.path.join(s1_dir, f"{field_id}_{date}_{band}_10m.tif")
                write_raster(path, train_arr)
        for date in s2_timestamps:
            s2_dir = os.path.join(train_imagery_s2_dir, field_id, date)
            os.makedirs(s2_dir, exist_ok=True)
            for band in s2_bands:
                train_arr = np.random.randint(dtype_max, size=(SIZE, SIZE), dtype=dtype)
                path = os.path.join(s2_dir, f"{field_id}_{date}_{band}_10m.tif")
                write_raster(path, train_arr)
        label_path = os.path.join(train_labels_dir, f"{field_id}.tif")
        label_arr = np.random.randint(9, size=(SIZE, SIZE), dtype=dtype)
        write_raster(label_path, label_arr)


if __name__ == "__main__":
    generate_test_data()
