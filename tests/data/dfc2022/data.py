#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio

from torchgeo.datasets import DFC2022

SIZE = 32

np.random.seed(0)


train_set = [
    {
        "image": "labeled_train/Nantes_Saint-Nazaire/BDORTHO/44-2013-0295-6713-LA93-0M50-E080.tif",  # noqa: E501
        "dem": "labeled_train/Nantes_Saint-Nazaire/RGEALTI/44-2013-0295-6713-LA93-0M50-E080_RGEALTI.tif",  # noqa: E501
        "target": "labeled_train/Nantes_Saint-Nazaire/UrbanAtlas/44-2013-0295-6713-LA93-0M50-E080_UA2012.tif",  # noqa: E501
    },
    {
        "image": "labeled_train/Nice/BDORTHO/06-2014-1007-6318-LA93-0M50-E080.tif",  # noqa: E501
        "dem": "labeled_train/Nice/RGEALTI/06-2014-1007-6318-LA93-0M50-E080_RGEALTI.tif",  # noqa: E501
        "target": "labeled_train/Nice/UrbanAtlas/06-2014-1007-6318-LA93-0M50-E080_UA2012.tif",  # noqa: E501
    },
]

unlabeled_set = [
    {
        "image": "unlabeled_train/Calais_Dunkerque/BDORTHO/59-2012-0650-7077-LA93-0M50-E080.tif",  # noqa: E501
        "dem": "unlabeled_train/Calais_Dunkerque/RGEALTI/59-2012-0650-7077-LA93-0M50-E080_RGEALTI.tif",  # noqa: E501
    },
    {
        "image": "unlabeled_train/LeMans/BDORTHO/72-2013-0469-6789-LA93-0M50-E080.tif",  # noqa: E501
        "dem": "unlabeled_train/LeMans/RGEALTI/72-2013-0469-6789-LA93-0M50-E080_RGEALTI.tif",  # noqa: E501
    },
]

val_set = [
    {
        "image": "val/Marseille_Martigues/BDORTHO/13-2014-0900-6268-LA93-0M50-E080.tif",  # noqa: E501
        "dem": "val/Marseille_Martigues/RGEALTI/13-2014-0900-6268-LA93-0M50-E080_RGEALTI.tif",  # noqa: E501
    },
    {
        "image": "val/Clermont-Ferrand/BDORTHO/63-2013-0711-6530-LA93-0M50-E080.tif",  # noqa: E501
        "dem": "val/Clermont-Ferrand/RGEALTI/63-2013-0711-6530-LA93-0M50-E080_RGEALTI.tif",  # noqa: E501
    },
]


def create_file(path: str, dtype: str, num_channels: int) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = "epsg:4326"
    profile["transform"] = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1)
    profile["height"] = SIZE
    profile["width"] = SIZE

    if "float" in profile["dtype"]:
        Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])
    else:
        Z = np.random.randint(
            np.iinfo(profile["dtype"]).max, size=(SIZE, SIZE), dtype=profile["dtype"]
        )

    with rasterio.open(path, "w", **profile) as src:
        for i in range(1, profile["count"] + 1):
            src.write(Z, i)


if __name__ == "__main__":
    for split in DFC2022.metadata:
        directory = DFC2022.metadata[split]["directory"]
        filename = DFC2022.metadata[split]["filename"]

        # Remove old data
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        if os.path.exists(filename):
            os.remove(filename)

        if split == "train":
            files = train_set
        elif split == "train-unlabeled":
            files = unlabeled_set
        else:
            files = val_set

        for file_dict in files:
            # Create image file
            path = file_dict["image"]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            create_file(path, dtype="uint8", num_channels=3)

            # Create DEM file
            path = file_dict["dem"]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            create_file(path, dtype="float32", num_channels=1)

            # Create mask file
            if split == "train":
                path = file_dict["target"]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                create_file(path, dtype="uint8", num_channels=1)

        # Compress data
        shutil.make_archive(filename.replace(".zip", ""), "zip", ".", directory)

        # Compute checksums
        with open(filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{filename}: {md5}")
