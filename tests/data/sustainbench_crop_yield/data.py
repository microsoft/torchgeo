#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np

SIZE = 32
NUM_SAMPLES = 3
NUM_BANDS = 9

np.random.seed(0)

countries = ["argentina", "brazil", "usa"]
splits = ["train", "dev", "test"]

root_dir = "soybeans"


def create_files(path: str, split: str) -> None:
    hist_img = np.random.random(size=(NUM_SAMPLES, SIZE, SIZE, NUM_BANDS))
    np.savez(os.path.join(path, f"{split}_hists.npz"), data=hist_img)

    target = np.random.random(size=(NUM_SAMPLES, 1))
    np.savez(os.path.join(path, f"{split}_yields.npz"), data=target)

    ndvi = np.random.random(size=(NUM_SAMPLES, SIZE))
    np.savez(os.path.join(path, f"{split}_ndvi.npz"), data=ndvi)

    year = np.array(["2009"] * NUM_SAMPLES, dtype="<U4")
    np.savez(os.path.join(path, f"{split}_years.npz"), data=year)


if __name__ == "__main__":
    # Remove old data
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)

    for country in countries:
        dir = os.path.join(root_dir, country)
        os.makedirs(dir)

        for split in splits:
            create_files(dir, split)

    filename = root_dir + ".zip"

    # Compress data
    shutil.make_archive(filename.replace(".zip", ""), "zip", ".", root_dir)

    # Compute checksums
    with open(filename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{filename}: {md5}")
