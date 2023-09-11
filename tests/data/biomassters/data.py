#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import rasterio

metadata_train = "The_BioMassters_-_features_metadata.csv.csv"

targets = "train_agbm.zip"

splits = ["train", "test"]

sample_ids = ["0003d2eb", "000aa810"]

months = ["September", "October", "November"]

satellite = ["S1", "S2"]


def create_S1_data():
    pass


def create_S2_data():
    pass


def create_metadata():
    pass


def create_target():
    pass


if __name__ == "__main__":
    for split in splits:
        for id in sample_ids:
            for sat in satellite:
                path = id + "_" + str(sat)
                for idx, month in enumerate(months):
                    file_path = path + "_" + f"{idx:02d}" + ".tif"

                    if sat == "S1":
                        create_S1_data(file_path)
                    else:
                        create_S2_data(file_path)

        # create target data
        if split == "train":
            create_target(id + "_agbm.tif")
