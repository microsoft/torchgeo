#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
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
    dtype_max = np.iinfo(dtype).max

    size = 36

    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

    profile = {
        "dtype": dtype,
        "width": size,
        "height": size,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(10.0, 0.0, 535840.0, 0.0, -10.0, 3079680.0),
    }

    folder_path = os.path.join(
        root, "tests", "data", "agrifieldnet", "ref_agrifieldnet_competition_v1"
    )

    train_img_dir = os.path.join(folder_path, "ref_agrifieldnet_competition_v1_source")
    train_mask_dir = os.path.join(
        folder_path, "ref_agrifieldnet_competition_v1_labels_train"
    )
    test_field_dir = os.path.join(
        folder_path, "ref_agrifieldnet_competition_v1_labels_test"
    )

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_field_dir, exist_ok=True)

    train_ids = ["00001", "00002", "00003"]
    test_ids = ["00001", "00002", "00003"]

    for i in range(n_samples):
        train_id = train_ids[i]
        train_patch_dir = os.path.join(
            train_img_dir, "ref_agrifieldnet_competition_v1_source_" + train_id
        )
        os.makedirs(train_patch_dir, exist_ok=True)

        for band in bands:
            train_arr = np.random.randint(dtype_max, size=(size, size), dtype=dtype)
            path = os.path.join(train_patch_dir, f"{band}.tif")
            with rasterio.open(path, "w", **profile) as src:
                src.write(train_arr, 1)

        train_mask_arr = np.random.randint(2, size=(size, size), dtype=np.uint16)
        train_mask_patch_dir = os.path.join(
            train_mask_dir, "ref_agrifieldnet_competition_v1_labels_train_" + train_id
        )
        os.makedirs(train_mask_patch_dir, exist_ok=True)
        path = os.path.join(train_mask_patch_dir, "raster_labels.tif")
        with rasterio.open(path, "w", **profile) as src:
            src.write(train_mask_arr, 1)

        train_field_arr = np.random.randint(2, size=(size, size), dtype=np.uint16)
        path = os.path.join(
            train_mask_dir,
            "ref_agrifieldnet_competition_v1_labels_train_" + train_id,
            "field_ids.tif",
        )
        with rasterio.open(path, "w", **profile) as src:
            src.write(train_field_arr, 1)

        test_id = test_ids[i]

        test_field_arr = np.random.randint(2, size=(size, size), dtype=np.uint16)
        test_field_patch_dir = os.path.join(
            test_field_dir, "ref_agrifieldnet_competition_v1_labels_test_" + test_id
        )
        os.makedirs(test_field_patch_dir, exist_ok=True)
        path = os.path.join(test_field_patch_dir, "field_ids.tif")
        with rasterio.open(path, "w", **profile) as src:
            src.write(test_field_arr, 1)

    # Create archive
    os.chdir(os.path.dirname(folder_path))
    shutil.make_archive(
        "ref_agrifieldnet_competition_v1",
        "gztar",
        root_dir=os.path.dirname(folder_path),
        base_dir="ref_agrifieldnet_competition_v1",
    )
    return calculate_md5(
        os.path.join(
            os.path.dirname(folder_path), "ref_agrifieldnet_competition_v1.tar.gz"
        )
    )


if __name__ == "__main__":
    md5_hash = generate_test_data(os.getcwd(), 3)
    print(md5_hash + "\n")
