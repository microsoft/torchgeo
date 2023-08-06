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

PATHS = [
    os.path.join(
        "FireRisk", "train", "High", "27032281_4_-103.430441201095_44.2804260315038.png"
    ),
    os.path.join(
        "FireRisk", "train", "Low", "27032391_2_-103.058289903541_44.3007203324261.png"
    ),
    os.path.join(
        "FireRisk",
        "train",
        "Moderate",
        "27033601_3_-98.95279624632_44.455109470962.png",
    ),
    os.path.join(
        "FireRisk",
        "train",
        "Non-burnable",
        "27033161_6_-100.447787439271_44.4136022778593.png",
    ),
    os.path.join(
        "FireRisk",
        "train",
        "Very_High",
        "27041631_5_-123.547051830273_41.5463004986268.png",
    ),
    os.path.join(
        "FireRisk", "val", "High", "35501951_4_-73.9911660056379_41.2755665931274.png"
    ),
    os.path.join(
        "FireRisk", "val", "Low", "35501621_2_-75.0371666057303_41.4540009148918.png"
    ),
    os.path.join(
        "FireRisk",
        "val",
        "Moderate",
        "35501731_3_-74.6879125510064_41.3954685534897.png",
    ),
    os.path.join(
        "FireRisk",
        "val",
        "Non-burnable",
        "35502061_6_-73.6436892181052_41.2142019946826.png",
    ),
    os.path.join(
        "FireRisk",
        "val",
        "Very_High",
        "35502941_5_-122.968467383602_40.2960022654498.png",
    ),
]


def create_file(path: str) -> None:
    Z = np.random.randint(255, size=(SIZE, SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(Z).convert("RGB")
    img.save(path)


if __name__ == "__main__":
    directory = "FireRisk"
    filename = "FireRisk.zip"

    # remove old data
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    for path in PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        create_file(path)

    # compress data
    shutil.make_archive(directory, "zip", ".", directory)

    # compute checksum
    with open(filename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{filename}: {md5}")
