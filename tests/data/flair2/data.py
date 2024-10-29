#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import sys
from typing import Sequence
import string

import numpy as np
from pyproj import CRS
import rasterio


# General hyperparams
IMG_SIZE = 512
DUMMY_DATA_SIZE = 10

# Directory structure
root_dir = "{0}/FLAIR2"
splits: Sequence[str] = ("train", "test")
dir_names: dict[dict[str, str]] = {
    "train": {
        "img": "flair_aerial_train",
        "sen": "flair_sen_train",
        "msk": "flair_labels_train",
    },
    "test": {
        "img": "flair_2_aerial_test",
        "sen": "flair_2_sen_test",
        "msk": "flair_2_labels_test",
    }
}
# Replace with random digits and letters
sub_sub_dir_format = "D{0}_{1}/Z{2}_{3}"


# Aerial specifics
aerial_all_bands: tuple = ("B01", "B02", "B03", "B04", "B05")
aerial_pixel_values: list[int] = list(range(256))
aerial_profile = {
    "dtype": np.uint8,
    "count": len(aerial_all_bands),
    "width": IMG_SIZE,
    "height": IMG_SIZE,
    "crs": CRS.from_epsg(2154),
    "driver": "GTiff"
}
aerial_format = ".tif"

# Sentinel specifics
sentinel_format = [".npy", ".npy", ".txt"]


# Label specifics
# Theoretically the labels are between 1 and 18, although the flair2 description groups all pixels >13 into one class
labels_pixel_values: list[int] = list(range(1, 19))
labels_profile = {
    "dtype": np.byte,
    "count": 1,
    "width": IMG_SIZE,
    "height": IMG_SIZE,
    # In theory the date is broken at the moment but assume correct for now
    "crs": CRS.from_epsg(2154),
    "driver": "GTiff"
}
labels_format = ".tif"


def populate_sub_sub_dirs(dir_path: str, rng: np.random.Generator, type: str) -> None:
    file_id = rng.integers(100, 1000)
    match type:
        case "img":
            create_aerial_image(dir_path, rng, file_id)
        case "sen":
            create_sentinel_arrays(dir_path, rng, file_id)
        case "msk":
            create_label_mask(dir_path, rng, file_id)
        case _:
            raise ValueError(f"Unknown type: {type}")


def create_aerial_image(dir_path: str, rng: np.random.Generator, id: int) -> None:
    with rasterio.open(os.path.join(dir_path, f"IMG_000{id}{aerial_format}"), 'w', **aerial_profile) as src:
        for i in range(len(aerial_all_bands)):
            data = rng.choice(aerial_pixel_values, size=(IMG_SIZE, IMG_SIZE), replace=True).astype(np.uint8)
            src.write(data, i+1)


def create_sentinel_arrays(dir_path: str, rng: np.random.Generator, id: int) -> None:
    # TODO: Implement
    pass


def create_label_mask(dir_path: str, rng: np.random.Generator, id: int) -> None:
    data = rng.choice(labels_pixel_values, size=(IMG_SIZE, IMG_SIZE), replace=True).astype(np.byte)
    with rasterio.open(os.path.join(dir_path, f"MSK_000{id}{labels_format}"), 'w', **labels_profile) as src:
        src.write(data, 1)


if __name__ == "__main__":
    root_dir = root_dir.format(os.getcwd())
    
    # Remove the root directory if it exists
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    # Create the directory structure
    for split in splits:
        for type, sub_dir in dir_names[split].items():
            for i in range(DUMMY_DATA_SIZE):
                # Reproducible and the same for all types
                rng: np.random.Generator = np.random.default_rng(seed=hash(f"{split}{i}")  % ((sys.maxsize + 1) * 2))
                
                random_domain = rng.integers(100, 999)
                random_year = rng.integers(2010, 2022)
                random_zone = rng.integers(10, 20)
                random_area = "".join(rng.choice(list(string.ascii_uppercase), size=2))
                
                # E.g. D123_2021/Z1_UF
                sub_sub_dir = sub_sub_dir_format.format(random_domain, random_year, random_zone, random_area)
                
                # type adds last directory, one of: img, sen, msk
                dir_path = os.path.join(root_dir, sub_dir, sub_sub_dir, type)
                os.makedirs(dir_path, exist_ok=True)
                
                populate_sub_sub_dirs(dir_path, rng, type)

    # zip and compute md5
    shutil.make_archive(root_dir, "zip", ".", root_dir)