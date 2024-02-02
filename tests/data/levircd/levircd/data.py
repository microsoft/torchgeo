#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import zipfile

import numpy as np
from PIL import Image

np.random.seed(0)


def create_image(path: str) -> None:
    Z = np.random.randint(255, size=(1, 1, 3), dtype=np.uint8)
    img = Image.fromarray(Z).convert("RGB")
    img.save(path)


def create_mask(path: str) -> None:
    Z = np.random.randint(2, size=(1, 1, 3), dtype=np.uint8) * 255
    img = Image.fromarray(Z).convert("L")
    img.save(path)


if __name__ == "__main__":
    splits = ["train", "val", "test"]
    filenames = ["train.zip", "val.zip", "test.zip"]
    directories = ["A", "B", "label"]

    for split, filename in zip(splits, filenames):
        for directory in directories:
            os.mkdir(directory)

        for i in range(2):
            path = os.path.join("A", f"{split}_{i}.png")
            create_image(path)

            path = os.path.join("B", f"{split}_{i}.png")
            create_image(path)

            path = os.path.join("label", f"{split}_{i}.png")
            create_mask(path)

        # compress data
        with zipfile.ZipFile(filename, mode="a") as f:
            for directory in directories:
                for file in os.listdir(directory):
                    f.write(os.path.join(directory, file))

        for directory in directories:
            shutil.rmtree(directory)

        # compute checksum
        with open(filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{filename}: {md5}")
