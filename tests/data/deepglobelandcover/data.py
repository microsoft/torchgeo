#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
from PIL import Image
from torchvision.datasets.utils import calculate_md5


def generate_test_data(root: str, n_samples: int = 3) -> str:
    """Create test data archive for DeepGlobeLandCover dataset.

    Args:
        root: path to store test data
        n_samples: number of samples.

    Returns:
        md5 hash of created archive
    """
    dtype = np.uint8
    size = 2

    folder_path = os.path.join(root, "data")

    train_img_dir = os.path.join(folder_path, "data", "training_data", "images")
    train_mask_dir = os.path.join(folder_path, "data", "training_data", "masks")
    test_img_dir = os.path.join(folder_path, "data", "test_data", "images")
    test_mask_dir = os.path.join(folder_path, "data", "test_data", "masks")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    train_ids = [1, 2, 3]
    test_ids = [8, 9, 10]

    for i in range(n_samples):
        train_id = train_ids[i]
        test_id = test_ids[i]

        dtype_max = np.iinfo(dtype).max
        train_arr = np.random.randint(dtype_max, size=(size, size, 3), dtype=dtype)
        train_img = Image.fromarray(train_arr)
        train_img.save(os.path.join(train_img_dir, str(train_id) + "_sat.jpg"))

        test_arr = np.random.randint(dtype_max, size=(size, size, 3), dtype=dtype)
        test_img = Image.fromarray(test_arr)
        test_img.save(os.path.join(test_img_dir, str(test_id) + "_sat.jpg"))

        train_mask_arr = np.full((size, size, 3), (0, 255, 255), dtype=dtype)
        train_mask_img = Image.fromarray(train_mask_arr)
        train_mask_img.save(os.path.join(train_mask_dir, str(train_id) + "_mask.png"))

        test_mask_arr = np.full((size, size, 3), (255, 0, 255), dtype=dtype)
        test_mask_img = Image.fromarray(test_mask_arr)
        test_mask_img.save(os.path.join(test_mask_dir, str(test_id) + "_mask.png"))

    # Create archive
    shutil.make_archive(folder_path, "zip", folder_path)
    shutil.rmtree(folder_path)
    return calculate_md5(f"{folder_path}.zip")


if __name__ == "__main__":
    md5_hash = generate_test_data(os.getcwd(), 3)
    print(md5_hash + "\n")
