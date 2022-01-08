#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import random

import rasterio
import numpy as np

from torchgeo.datasets import Urban3DChallenge

ROOT = "/path/to/dataset/"
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
    for split in Urban3DChallenge.directories:
        ds = Urban3DChallenge(ROOT, split=split)

        directory = Urban3DChallenge.directories[split]

        # Remove old data
        if os.path.isdir(directory):
            shutil.rmtree(directory)

        files = random.sample(ds.files, k=NUM_SAMPLES)
        for file_dict in files:
            # Create image file
            path = file_dict["rgb"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)

            # Create DTM file
            path = file_dict["dtm"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)

            # Create DTM file
            path = file_dict["dsm"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)

            # Create semantic mask file
            path = file_dict["binary_mask"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)

            # Create instance mask file
            path = file_dict["instance_mask"]
            output_path = path.replace(ROOT, "")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_file(path, output_path)
