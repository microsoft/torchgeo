#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import random
import hashlib

import rasterio
import numpy as np

from torchgeo.datasets import DFC2022

ROOT = "/mnt/e/data/dfc2022/"
NUM_SAMPLES = 2
SIZE = 32

np.random.seed(0)
random.seed(0)


def create_file(path: str, output_path: str) -> None:
    src = rasterio.open(path)
    profile = src.profile
    dtype = profile["dtype"]
    profile["height"] = SIZE
    profile["width"] = SIZE
    max_val = np.finfo(dtype).max if "float" in dtype else np.iinfo(dtype).max

    if "float" in dtype:
        Z = np.random.randn(SIZE, SIZE).astype(dtype)
    else:
        Z = np.random.randint(max_val, size=(SIZE, SIZE), dtype=dtype)

    dst = rasterio.open(output_path, "w", **profile)
    for i in range(1, profile["count"] + 1):
        dst.write(Z, i)


if __name__ == "__main__":
    for split in DFC2022.metadata:
        ds = DFC2022(ROOT, split=split)

        directory = DFC2022.metadata[split]["directory"]
        filename = DFC2022.metadata[split]["filename"]

        # Remove old data
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        if os.path.exists(filename):
            os.remove(filename)

        files = random.sample(ds.files, k=NUM_SAMPLES)
        for file_dict in files:
            # Create image file
            path = file_dict["image"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)

            # Create DEM file
            path = file_dict["dem"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)

            # Create mask file
            if split == "train":
                path = file_dict["target"]
                output_path = path.replace(ROOT, "")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                create_file(path, output_path)

        # Compress data
        shutil.make_archive(filename.replace(".zip", ""), "zip", ".", directory)

        # Compute checksums
        with open(filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{filename}: {md5}")
