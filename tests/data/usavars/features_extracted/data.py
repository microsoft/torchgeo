#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

splits = ["train", "val", "test"]
csv_file_name = "{}_usa_vars.csv"
all_labels = ["treecover", "elevation", "population"]
feature_extractors = ["rcf", "resnet18"]
valid_splits = ["train", "val", "test"]

tile_names = {
    "train": ["tile_1318,1204.tif", "tile_171,1864.tif", "tile_2157,1929.tif"],
    "val": ["tile_1425,404.tif", "tile_1359,1493.tif", "tile_385,999.tif"],
    "test": ["tile_664,1724.tif", "tile_478,4085.tif", "tile_128,1870.tif"],
}


if __name__ == "__main__":
    for feature_extractor in feature_extractors:
        df = pd.DataFrame()
        split_data = []
        for split in valid_splits:
            num_ds_points = len(tile_names[split])
            features = np.random.randn(num_ds_points, 512)

            split_df = pd.DataFrame(features)
            split_df["centroid_lat"] = np.random.randn(num_ds_points)
            split_df["centroid_lon"] = np.random.randn(num_ds_points)
            split_df["filename"] = tile_names[split]
            split_df["split"] = split

            for label in all_labels:
                split_df[label] = np.random.randn(num_ds_points)

            df = pd.concat([df, split_df])

        df.to_csv(csv_file_name.format(feature_extractor))
