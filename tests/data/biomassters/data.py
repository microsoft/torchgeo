#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import hashlib
import os
import shutil

import numpy as np
import rasterio

metadata_train = "The_BioMassters_-_features_metadata.csv.csv"

csv_columns = [
    "filename",
    "chip_id",
    "satellite",
    "split",
    "month",
    "size",
    "cksum",
    "s3path_us",
    "s3path_eu",
    "s3path_as",
    "corresponding_agbm",
]

targets = "train_agbm.zip"

splits = ["train", "test"]

sample_ids = ["0003d2eb", "000aa810"]

months = ["September", "October", "November"]

satellite = ["S1", "S2"]

SIZE = 32


def create_tif_file(path: str, num_channels: int, dtype: str) -> None:
    """Create S1 or S2 data with num channels.

    Args:
        path: path where to save tif
        num_channels: number of channels (4 for S1, 11 for S2)
        dtype: uint16 for image data and float 32 for target
    """
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = "epsg:4326"
    profile["transform"] = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1)
    profile["height"] = SIZE
    profile["width"] = SIZE
    profile["compress"] = "lzw"
    profile["predictor"] = 2

    if "float" in profile["dtype"]:
        Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])
    else:
        Z = np.random.randint(
            np.iinfo(profile["dtype"]).max, size=(SIZE, SIZE), dtype=profile["dtype"]
        )

    with rasterio.open(path, "w", **profile) as src:
        for i in range(1, profile["count"] + 1):
            src.write(Z, i)


# filename,chip_id,satellite,split,month,size,cksum,s3path_us,s3path_eu,s3path_as,corresponding_agbm
if __name__ == "__main__":
    csv_rows = []
    for split in splits:
        os.makedirs(f"{split}_features", exist_ok=True)
        if split == "train":
            os.makedirs("train_agbm", exist_ok=True)
        for id in sample_ids:
            for sat in satellite:
                path = id + "_" + str(sat)
                for idx, month in enumerate(months):
                    # S2 data is not present for every month
                    if sat == "S2" and idx == 1:
                        continue
                    file_path = path + "_" + f"{idx:02d}" + ".tif"

                    csv_rows.append(
                        [
                            file_path,
                            id,
                            sat,
                            split,
                            month,
                            "0",
                            "0",
                            "path",
                            "path",
                            "path",
                            id + "_agbm.tif",
                        ]
                    )

                    # file path to save
                    file_path = os.path.join(f"{split}_features", file_path)

                    if sat == "S1":
                        create_tif_file(file_path, num_channels=4, dtype="uint16")
                    else:
                        create_tif_file(file_path, num_channels=11, dtype="uint16")

            # create target data one per id
            if split == "train":
                create_tif_file(
                    os.path.join(f"{split}_agbm", id + "_agbm.tif"),
                    num_channels=1,
                    dtype="float32",
                )

    # write out metadata

    with open(metadata_train, "w") as csv_file:
        wr = csv.writer(csv_file)
        wr.writerow(csv_columns)
        for row in csv_rows:
            wr.writerow(row)

    # zip up feature and target folders
    zip_dirs = ["train_features", "test_features", "train_agbm"]
    for dir in zip_dirs:
        shutil.make_archive(dir, "zip", dir)
        # Compute checksums
        with open(dir + ".zip", "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{dir}: {md5}")
