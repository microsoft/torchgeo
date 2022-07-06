#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine
from torchvision.datasets.utils import calculate_md5


def write_data(
    path: str, img: np.ndarray, driver: str, crs: CRS, transform: Affine
) -> None:
    with rio.open(
        path,
        "w",
        driver=driver,
        height=img.shape[0],
        width=img.shape[1],
        count=3,
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(1, dst.count + 1):
            dst.write(img, i)


def generate_test_data(root: str, n_samples: int = 2) -> str:
    """Creates test data archive for InriaAerialImageLabeling dataset and
    returns its md5 hash.

    Args:
        root (str): Path to store test data
        n_samples (int, optional): Number of samples. Defaults to 2.

    Returns:
        str: md5 hash of created archive
    """
    dtype = np.dtype("uint8")
    size = (8, 8)

    driver = "GTiff"
    transform = Affine(0.3, 0.0, 616500.0, 0.0, -0.3, 3345000.0)
    crs = CRS.from_epsg(26914)

    folder_path = os.path.join(root, "AerialImageDataset")

    img_dir = os.path.join(folder_path, "train", "images")
    lbl_dir = os.path.join(folder_path, "train", "gt")
    timg_dir = os.path.join(folder_path, "test", "images")

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(lbl_dir):
        os.makedirs(lbl_dir)
    if not os.path.exists(timg_dir):
        os.makedirs(timg_dir)

    for i in range(n_samples):

        dtype_max = np.iinfo(dtype).max
        img = np.random.randint(dtype_max, size=size, dtype=dtype)
        lbl = np.random.randint(2, size=size, dtype=dtype)
        timg = np.random.randint(dtype_max, size=size, dtype=dtype)

        img_path = os.path.join(img_dir, f"austin{i+1}.tif")
        lbl_path = os.path.join(lbl_dir, f"austin{i+1}.tif")
        timg_path = os.path.join(timg_dir, f"austin{i+10}.tif")

        write_data(img_path, img, driver, crs, transform)
        write_data(lbl_path, lbl, driver, crs, transform)
        write_data(timg_path, timg, driver, crs, transform)

    # Create archive
    archive_path = os.path.join(root, "NEW2-AerialImageDataset")
    shutil.make_archive(
        archive_path, "zip", root_dir=root, base_dir="AerialImageDataset"
    )
    shutil.rmtree(folder_path)
    return calculate_md5(f"{archive_path}.zip")


if __name__ == "__main__":
    md5_hash = generate_test_data(os.getcwd(), 5)
    print(md5_hash)
