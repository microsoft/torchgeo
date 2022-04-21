#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil
import tarfile
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.transform import Affine

SIZE = 32
EPSGCODE = 32736

np.random.seed(0)

band_filenames = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B11.tif",
    "B12.tif",
    "CLD.tif",
]

# share same dummy input imagery for all datasets
image_root = "ref_african_crops_datasets"

image_directories = [
    {
        "path": "ref_african_crops_datasets_source_00_20180412",
        "bbox": [
            34.18191992149459,
            0.4724181558451209,
            34.2766208946704,
            0.5894898556277989,
        ],
        "datetime": "2018-04-12T00:00:00Z",
    },
    {
        "path": "ref_african_crops_datasets_source_00_20180606",
        "bbox": [
            34.18191992149459,
            0.4724181558451209,
            34.2766208946704,
            0.5894898556277989,
        ],
        "datetime": "2018-06-06T00:00:00Z",
    },
    {
        "path": "ref_african_crops_datasets_source_01_20180412",
        "bbox": [
            34.2768887802958,
            0.551917324714215,
            34.3368545919656,
            0.567391591348043,
        ],
        "datetime": "2018-04-12T00:00:00Z",
    },
    {
        "path": "ref_african_crops_datasets_source_01_20180606",
        "bbox": [
            34.2768887802958,
            0.551917324714215,
            34.3368545919656,
            0.567391591348043,
        ],
        "datetime": "2018-06-06T00:00:00Z",
    },
]

# each child class has its own labels under "properties"
# but shares the same geometry coordinates
# to match the input imagery

# labels for CropTypeKenyaPlantVillage
kenya_label_root = "ref_african_crops_kenya_01_labels"
kenya_label_dir = [
    {
        "path": "ref_african_crops_kenya_01_labels_00",
        "num_features": 2,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [635776.2598575517, 59518.90794878066],
                    [635798.7265667534, 59512.93407433634],
                    [635778.4349066955, 59475.19789931116],
                    [635762.3879757895, 59483.20071830105],
                    [635776.2598575517, 59518.90794878066],
                ]
            ],
        },
        "properties": {"Crop1": "Cassava"},
    },
    {
        "path": "ref_african_crops_kenya_01_labels_01",
        "num_features": 0,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [644386.1143698449, 62530.949496365654],
                    [644350.5396486986, 62527.08096165654],
                    [644348.5983758575, 62544.989854711785],
                    [644382.9725918629, 62543.27296036849],
                    [644386.1143698449, 62530.949496365654],
                ]
            ],
        },
        "properties": {"Crop": "Cassava"},
    },
]

# labels for CropTypeTanzaniaGAFCO
tanzania_label_root = "ref_african_crops_tanzania_01_labels"
tanzania_label_dir = [
    {
        "path": "ref_african_crops_tanzania_01_labels_00",
        "num_features": 2,
        "geometry": kenya_label_dir[0]["geometry"],
        "properties": {"Crop1": "Dry Bean"},
    },
    {
        "path": "ref_african_crops_tanzania_01_labels_01",
        "num_features": 0,
        "geometry": kenya_label_dir[0]["geometry"],
        "properties": {"Crop1": "Dry Bean"},
    },
]

# labels for CropTypeUgandaDalbergDataInsight
uganda_label_root = "ref_african_crops_uganda_01_labels"
uganda_label_dir = [
    {
        "path": "ref_african_crops_uganda_01_labels_00",
        "num_features": 2,
        "geometry": kenya_label_dir[0]["geometry"],
        "properties": {"crop1": "sorghum"},
    },
    {
        "path": "ref_african_crops_uganda_01_labels_01",
        "num_features": 0,
        "geometry": kenya_label_dir[0]["geometry"],
        "properties": {"crop1": "sorghum"},
    },
]


def create_imagery(path: str, dtype: str, num_channels: int) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = f"epsg:{EPSGCODE}"
    profile["transform"] = Affine(
        9.997558351553115,
        0.0,
        642068.9738940755,
        0.0,
        -9.999449954310554,
        65607.7835340335,
    )
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

    src = rasterio.open(path, "w", **profile)
    for i in range(1, profile["count"] + 1):
        src.write(Z, i)


def create_stac_imagery(path: str, bbox: List, date: str) -> None:
    image_stac = {
        "bbox": bbox,
        "properties": {"constellation": "Sentinel-2", "datetime": date},
    }

    with open(path, "w") as f:
        json.dump(image_stac, f)


def create_label(
    path: str, geometry: Dict, properties: Dict, num_features: int = 1
) -> None:
    feature = {"type": "Feature", "properties": properties, "geometry": geometry}

    label_data = {
        "type": "FeatureCollection",
        "features": [feature for i in range(num_features)],
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{EPSGCODE}"},
        },
    }

    with open(path, "w") as f:
        json.dump(label_data, f)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


if __name__ == "__main__":
    if os.path.isdir(image_root):
        shutil.rmtree(image_root)

    # create common imagery used by all sub datasets
    for directory in image_directories:
        os.makedirs(os.path.join(image_root, directory["path"]), exist_ok=True)

        # create separate bands
        for f in band_filenames:
            file_path = os.path.join(image_root, directory["path"], f)
            create_imagery(path=file_path, dtype="int32", num_channels=1)

        # create stac.json
        create_stac_imagery(
            os.path.join(image_root, directory["path"], "stac.json"),
            directory["bbox"],
            directory["datetime"],
        )

    # create label and corresponding stac.sjon
    label_roots = [kenya_label_root, tanzania_label_root, uganda_label_root]
    label_directories = [kenya_label_dir, tanzania_label_dir, uganda_label_dir]

    for root, label_dirs in zip(label_roots, label_directories):
        if os.path.isdir(root):
            shutil.rmtree(root)

        for label_dir in label_dirs:
            label_dir_path = os.path.join(root, label_dir["path"])
            os.makedirs(label_dir_path, exist_ok=True)
            create_label(
                path=os.path.join(label_dir_path, "labels.geojson"),
                num_features=label_dir["num_features"],
                geometry=label_dir["geometry"],
                properties=label_dir["properties"],
            )

    # tar directories to .tar.gz and compute checksum
    for root in label_roots + [image_root]:
        output_filename = root + ".tar.gz"
        make_tarfile(output_filename, root)
        # Compute checksums
        with open(output_filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{root}: {md5}")
