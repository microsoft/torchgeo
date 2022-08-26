#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import hashlib
import os
import shutil
from typing import List

import numpy as np
from PIL import Image

SIZE = 32

np.random.seed(0)

PATHS = {
    "images": [
        "tiles/Site1/Site1_RGB_0_0_0_4000_4000.png",
        "tiles/Site2/Site2_RGB_0_0_0_4000_4000.png",
    ],
    "annotation": "mapping/final_dataset.csv",
}


def create_annotation(path: str, img_paths: List[str]) -> None:
    cols = ["img_path", "xmin", "ymin", "xmax", "ymax", "group", "AGB"]
    data = []
    for img_path in img_paths:
        data.append(
            [os.path.basename(img_path), 0, 0, SIZE / 2, SIZE / 2, "banana", 6.75]
        )
        data.append(
            [os.path.basename(img_path), SIZE / 2, SIZE / 2, SIZE, SIZE, "cacao", 6.75]
        )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(data)


def create_img(path: str) -> None:
    Z = np.random.rand(SIZE, SIZE, 3) * 255
    img = Image.fromarray(Z.astype("uint8")).convert("RGB")
    img.save(path)


if __name__ == "__main__":
    data_root = "reforesTree"

    # remove old data
    if os.path.isdir(data_root):
        shutil.rmtree(data_root)

    # create imagery
    for path in PATHS["images"]:
        os.makedirs(os.path.join(data_root, os.path.dirname(path)), exist_ok=True)
        create_img(os.path.join(data_root, path))

    # create annotations
    os.makedirs(
        os.path.join(data_root, os.path.dirname(PATHS["annotation"])), exist_ok=True
    )
    create_annotation(os.path.join(data_root, PATHS["annotation"]), PATHS["images"])

    # compress data
    shutil.make_archive(data_root, "zip", data_root)

    # Compute checksums
    with open(data_root + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{data_root}: {md5}")
