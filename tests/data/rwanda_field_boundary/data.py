#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio

dates = ("2021_03", "2021_04", "2021_08", "2021_10", "2021_11", "2021_12")
all_bands = ("B01", "B02", "B03", "B04")

SIZE = 32
NUM_SAMPLES = 5
np.random.seed(0)


def create_mask(fn: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0.0,
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": "epsg:3857",
        "compress": "lzw",
        "predictor": 2,
        "transform": rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        "blockysize": 32,
        "tiled": False,
        "interleave": "band",
    }
    with rasterio.open(fn, "w", **profile) as f:
        f.write(np.random.randint(0, 2, size=(SIZE, SIZE), dtype=np.uint8), 1)


def create_img(fn: str) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0.0,
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "crs": "epsg:3857",
        "compress": "lzw",
        "predictor": 2,
        "blockysize": 16,
        "transform": rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        "tiled": False,
        "interleave": "band",
    }
    with rasterio.open(fn, "w", **profile) as f:
        f.write(np.random.randint(0, 2, size=(SIZE, SIZE), dtype=np.uint16), 1)


if __name__ == "__main__":
    # Train and test images
    for split in ("train", "test"):
        for i in range(NUM_SAMPLES):
            for date in dates:
                directory = os.path.join(
                    f"nasa_rwanda_field_boundary_competition_source_{split}",
                    f"nasa_rwanda_field_boundary_competition_source_{split}_{i:02d}_{date}",  # noqa: E501
                )
                os.makedirs(directory, exist_ok=True)
                for band in all_bands:
                    create_img(os.path.join(directory, f"{band}.tif"))

        # Create collections.json, this isn't used by the dataset but is checked to
        # exist
        with open(
            f"nasa_rwanda_field_boundary_competition_source_{split}/collections.json",
            "w",
        ) as f:
            f.write("Not used")

    # Train labels
    for i in range(NUM_SAMPLES):
        directory = os.path.join(
            "nasa_rwanda_field_boundary_competition_labels_train",
            f"nasa_rwanda_field_boundary_competition_labels_train_{i:02d}",
        )
        os.makedirs(directory, exist_ok=True)
        create_mask(os.path.join(directory, "raster_labels.tif"))

    # Create directories and compute checksums
    for filename in [
        "nasa_rwanda_field_boundary_competition_source_train",
        "nasa_rwanda_field_boundary_competition_source_test",
        "nasa_rwanda_field_boundary_competition_labels_train",
    ]:
        shutil.make_archive(filename, "gztar", ".", filename)
        # Compute checksums
        with open(f"{filename}.tar.gz", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{filename}: {md5}")
