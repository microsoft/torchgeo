#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
from PIL import Image

SIZE = 32

np.random.seed(0)

PATHS = {
    "train": [
        os.path.join(
            "train", "agriculture_land", "grassland", "meadow", "P0115918.jpg"
        ),
        os.path.join("train", "water_area", "beach", "P0060208.jpg"),
    ],
    "test": [
        os.path.join("test", "agriculture_land", "grassland", "meadow", "P0115918.jpg"),
        os.path.join("test", "water_area", "beach", "P0060208.jpg"),
    ],
}


def create_file(path: str) -> None:
    Z = np.random.rand(SIZE, SIZE, 3) * 255
    img = Image.fromarray(Z.astype("uint8")).convert("RGB")
    img.save(path)


if __name__ == "__main__":
    for split, paths in PATHS.items():
        # remove old data
        if os.path.isdir(split):
            shutil.rmtree(split)
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            create_file(path)

        # compress data
        shutil.make_archive(split, "zip", ".", split)

        # Compute checksums
        with open(split + ".zip", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{split}: {md5}")
