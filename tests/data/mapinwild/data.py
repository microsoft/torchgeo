#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32

np.random.seed(0)

meta = {
    "driver": "GTiff",
    "nodata": None,
    "width": SIZE,
    "height": SIZE,
    "crs": CRS.from_epsg(32720),
    "transform": Affine(10.0, 0.0, 612190.0, 0.0, -10.0, 7324250.0),
}

count = {
    "ESA_WC": 1,
    "VIIRS": 1,
    "mask": 1,
    "s1_part1": 2,
    "s1_part2": 2,
    "s2_temporal_subset_part1": 10,
    "s2_temporal_subset_part2": 10,
    "s2_autumn_part1": 10,
    "s2_autumn_part2": 10,
    "s2_spring_part1": 10,
    "s2_spring_part2": 10,
    "s2_summer_part1": 10,
    "s2_summer_part2": 10,
    "s2_winter_part1": 10,
    "s2_winter_part2": 10,
}
dtype = {
    "ESA_WC": np.uint8,
    "VIIRS": np.float32,
    "mask": np.byte,
    "s1_part1": np.float64,
    "s1_part2": np.float64,
    "s2_temporal_subset_part1": np.uint16,
    "s2_temporal_subset_part2": np.uint16,
    "s2_autumn_part1": np.uint16,
    "s2_autumn_part2": np.uint16,
    "s2_spring_part1": np.uint16,
    "s2_spring_part2": np.uint16,
    "s2_summer_part1": np.uint16,
    "s2_summer_part2": np.uint16,
    "s2_winter_part1": np.uint16,
    "s2_winter_part2": np.uint16,
}
stop = {
    "ESA_WC": np.iinfo(np.uint8).max,
    "VIIRS": np.finfo(np.float32).max,
    "mask": np.iinfo(np.byte).max,
    "s1_part1": np.finfo(np.float64).max,
    "s1_part2": np.finfo(np.float64).max,
    "s2_temporal_subset_part1": np.iinfo(np.uint16).max,
    "s2_temporal_subset_part2": np.iinfo(np.uint16).max,
    "s2_autumn_part1": np.iinfo(np.uint16).max,
    "s2_autumn_part2": np.iinfo(np.uint16).max,
    "s2_spring_part1": np.iinfo(np.uint16).max,
    "s2_spring_part2": np.iinfo(np.uint16).max,
    "s2_summer_part1": np.iinfo(np.uint16).max,
    "s2_summer_part2": np.iinfo(np.uint16).max,
    "s2_winter_part1": np.iinfo(np.uint16).max,
    "s2_winter_part2": np.iinfo(np.uint16).max,
}

folder_path = os.path.join(os.getcwd(), "tests", "data", "mapinwild")

dict_all = {
    "s2_sum": ["s2_summer_part1", "s2_summer_part2"],
    "s2_spr": ["s2_spring_part1", "s2_spring_part2"],
    "s2_win": ["s2_winter_part1", "s2_winter_part2"],
    "s2_aut": ["s2_autumn_part1", "s2_autumn_part2"],
    "s1": ["s1_part1", "s1_part2"],
    "s2_temp": ["s2_temporal_subset_part1", "s2_temporal_subset_part2"],
}

md5s = {}
keys = count.keys()
modality_download_list = list(count.keys())

for source in modality_download_list:
    directory = os.path.join(folder_path, source)

    # Remove old data
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

    # Random images
    for i in range(1, 3):
        filename = f"{i}.tif"
        filepath = os.path.join(directory, filename)

        meta["count"] = count[source]
        meta["dtype"] = dtype[source]
        with rasterio.open(filepath, "w", **meta) as f:
            for j in range(1, count[source] + 1):
                if meta["dtype"] is np.float32 or meta["dtype"] is np.float64:
                    data = np.random.randn(SIZE, SIZE).astype(dtype[source])

                else:
                    data = np.random.randint(stop[source], size=(SIZE, SIZE)).astype(
                        dtype[source]
                    )
                f.write(data, j)

# Mimic the two-part structure of the dataset
for key in dict_all.keys():
    path_list = dict_all[key]
    path_list_dir_p1 = os.path.join(folder_path, path_list[0])
    path_list_dir_p2 = os.path.join(folder_path, path_list[1])
    n_ims = len(os.listdir(path_list_dir_p1))

    p1_list = os.listdir(path_list_dir_p1)
    p2_list = os.listdir(path_list_dir_p2)

    fh_idx = np.arange(0, n_ims / 2, dtype=int)
    sh_idx = np.arange(n_ims / 2, n_ims, dtype=int)

    for idx in sh_idx:
        sh_del = os.path.join(path_list_dir_p1, p1_list[idx])
        os.remove(sh_del)

    for idx in fh_idx:
        fh_del = os.path.join(path_list_dir_p2, p2_list[idx])
        os.remove(fh_del)

for i, source in zip(keys, modality_download_list):
    directory = os.path.join(folder_path, source)
    root = os.path.dirname(directory)

    # Compress data
    shutil.make_archive(directory, "zip", root_dir=root, base_dir=source)

    # Compute checksums
    with open(directory + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{directory}: {md5}")
        name = i + ".zip"
        md5s[name] = md5

tvt_split = pd.DataFrame(
    [["1", "2", "3"], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    index=["0", "1", "2"],
    columns=["train", "validation", "test"],
)
tvt_split.dropna()
tvt_split.to_csv(os.path.join(folder_path, "split_IDs.csv"))

with open(os.path.join(folder_path, "split_IDs.csv"), "rb") as f:
    csv_md5 = hashlib.md5(f.read()).hexdigest()
