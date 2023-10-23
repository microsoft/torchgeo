#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from torchvision.datasets.utils import calculate_md5


def generate_test_data(root: str) -> str:
    """Create test data archive for AgriFieldNet dataset.

    Args:
        root: path to store test data
        n_samples: number of samples.

    Returns:
        md5 hash of created archive
    """
    dtype = np.uint8
    dtype_max = np.iinfo(dtype).max

    size = 36

    bands = (
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

    classes = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36]).astype(
        np.uint16
    )

    profile = {
        "dtype": dtype,
        "width": size,
        "height": size,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(10.0, 0.0, 535840.0, 0.0, -10.0, 3079680.0),
    }

    dir = os.path.join(
        root, "tests", "data", "agrifieldnet"
    )

    train_img_dir = os.path.join(dir, "ref_agrifieldnet_competition_v1_source")
    train_mask_dir = os.path.join(dir, "ref_agrifieldnet_competition_v1_labels_train")
    test_field_dir = os.path.join(dir, "ref_agrifieldnet_competition_v1_labels_test")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_field_dir, exist_ok=True)

    source_tiles = ["00001", "00002", "00003", "00004", "00005"]
    train_tiles = ["00001", "00002", "00003", "00004"]
    test_tiles = ["00002", "00003", "00004", "00005"]

    for source_tile in source_tiles:
        directory = os.path.join(
            train_img_dir, "ref_agrifieldnet_competition_v1_source_" + source_tile
        )
        os.makedirs(directory, exist_ok=True)

        for band in bands:
            train_arr = np.random.randint(dtype_max, size=(size, size), dtype=dtype)
            path = os.path.join(directory, f"{band}.tif")
            with rasterio.open(path, "w", **profile) as src:
                src.write(train_arr, 1)

    for train_tile in train_tiles:
        train_mask_arr = np.random.choice(a=classes, size=(size, size))
        directory = os.path.join(
            train_mask_dir, "ref_agrifieldnet_competition_v1_labels_train_" + train_tile
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "raster_labels.tif")
        with rasterio.open(path, "w", **profile) as src:
            src.write(train_mask_arr, 1)

        train_field_arr = np.random.randint(20, size=(size, size), dtype=np.uint16)
        path = os.path.join(
            train_mask_dir,
            "ref_agrifieldnet_competition_v1_labels_train_" + train_tile,
            "field_ids.tif",
        )
        with rasterio.open(path, "w", **profile) as src:
            src.write(train_field_arr, 1)

    for test_tile in test_tiles:
        test_field_arr = np.random.randint(10, 30, size=(size, size), dtype=np.uint16)
        test_field_patch_dir = os.path.join(
            test_field_dir, "ref_agrifieldnet_competition_v1_labels_test_" + test_tile
        )
        os.makedirs(test_field_patch_dir, exist_ok=True)
        path = os.path.join(test_field_patch_dir, "field_ids.tif")
        with rasterio.open(path, "w", **profile) as src:
            src.write(test_field_arr, 1)

    # Create archive
    os.chdir(dir)
    shutil.make_archive(
        "ref_agrifieldnet_competition_v1_source",
        "gztar",
        root_dir=dir,
        base_dir="ref_agrifieldnet_competition_v1_source",
    )
    shutil.make_archive(
        "ref_agrifieldnet_competition_v1_labels_train",
        "gztar",
        root_dir=dir,
        base_dir="ref_agrifieldnet_competition_v1_labels_train",
    )
    shutil.make_archive(
        "ref_agrifieldnet_competition_v1_labels_test",
        "gztar",
        root_dir=dir,
        base_dir="ref_agrifieldnet_competition_v1_labels_test",
    )
    return None
    # return calculate_md5(
    #     os.path.join(os.path.dirname(dir), "ref_agrifieldnet_competition_v1.tar.gz"))


if __name__ == "__main__":
    md5_hash = generate_test_data(os.getcwd())
    print(md5_hash + "\n")
