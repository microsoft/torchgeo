#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil
import tarfile
from typing import List

import numpy as np
import rasterio
from rasterio.transform import Affine

SIZE = 32

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

root_image_dir = "ref_african_crops_tanzania_01_source"

image_directories = [
    {
        "path": "ref_african_crops_tanzania_01_source_00_20180102",
        "bbox": [
            33.568404763042174,
            -3.020344843124805,
            33.6664699555098,
            -2.9259331588640256,
        ],
        "datetime": "2018-01-02T00:00:00Z",
    },
    {
        "path": "ref_african_crops_tanzania_01_source_00_20180318",
        "bbox": [
            33.568404763042174,
            -3.020344843124805,
            33.6664699555098,
            -2.9259331588640256,
        ],
        "datetime": "2018-03-18T00:00:00Z",
    },
    {
        "path": "ref_african_crops_tanzania_01_source_01_20180102",
        "bbox": [
            33.568404763042174,
            -3.020344843124805,
            33.6664699555098,
            -2.9259331588640256,
        ],
        "datetime": "2018-01-02T00:00:00Z",
    },
    {
        "path": "ref_african_crops_tanzania_01_source_01_20180318",
        "bbox": [
            33.568404763042174,
            -3.020344843124805,
            33.6664699555098,
            -2.9259331588640256,
        ],
        "datetime": "2018-03-18T00:00:00Z",
    },
]

root_label_dir = "ref_african_crops_tanzania_01_labels"
label_directories = [
    {"path": "ref_african_crops_tanzania_01_labels_00", "num_features": 2},
    {"path": "ref_african_crops_tanzania_01_labels_01", "num_features": 0},
]


def create_imagery(path: str, dtype: str, num_channels: int) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = "epsg:32736"
    profile["transform"] = Affine(
        9.998199740032945,
        0.0,
        782853.9107002986,
        0.0,
        -10.002868812795558,
        9773930.117133377,
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
        "id": "ref_african_crops_tanzania_01_source_00_20180102",
        "properties": {"constellation": "Sentinel-2", "datetime": date},
    }

    with open(path, "w") as f:
        json.dump(image_stac, f)


def create_stac_labels(path: str) -> None:
    label_stac = {
        "assets": {
            "documentation": {
                "href": "../_common/documentation.pdf",
                "title": "Dataset Documentation",
                "type": "application/pdf",
            },
            "labels": {
                "href": "labels.geojson",
                "title": "Crop Labels",
                "type": "image/tiff",
            },
            "property_descriptions": {
                "href": "../_common/property_descriptions.csv",
                "title": "Label Property Descriptions",
                "type": "text/csv",
            },
        },
        "bbox": [
            33.568404763042174,
            -3.020344843124805,
            33.6664699555098,
            -2.9259331588640256,
        ],
        "collection": "ref_african_crops_tanzania_01_labels",
        "geometry": {
            "coordinates": [
                [
                    [33.6664699555098, -3.020344843124805],
                    [33.6664699555098, -2.9259331588640256],
                    [33.568404763042174, -2.9259331588640256],
                    [33.568404763042174, -3.020344843124805],
                    [33.6664699555098, -3.020344843124805],
                ]
            ],
            "type": "Polygon",
        },
        "id": "ref_african_crops_tanzania_01_labels_00",
        "properties": {
            "datetime": "2018-07-01T00:00:00Z",
            "label:description": "Tanzania Tile 00 Labels",
            "label:methods": ["manual"],
            "label:properties": "null",
            "label:tasks": ["classification"],
            "label:type": "vector",
        },
        "stac_extensions": ["label"],
        "stac_version": "1.0.0-beta.2",
        "type": "Feature",
    }

    with open(path, "w") as f:
        json.dump(label_stac, f)


def create_label(path: str, num_features: int = 1) -> None:
    feature = {
        "type": "Feature",
        "properties": {
            "Village": "Mwatumbe",
            "Region": "Simiyu",
            "Plot Area (acre)": 2,
            "Planting Date": "2018-03-30",
            "Estimated Harvest Date": "2018-10-30",
            "Crop": "Sunflower",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [564105.3968795359, 9676577.1790148],
                    [564158.887044993, 9676577.151513517],
                    [564158.8546518659, 9676514.179071717],
                    [564105.364513419, 9676514.206578337],
                    [564105.3968795359, 9676577.1790148],
                ]
            ],
        },
    }

    label_data = {
        "type": "FeatureCollection",
        "features": [feature for i in range(num_features)],
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32736"}},
    }

    with open(path, "w") as f:
        json.dump(label_data, f)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


if __name__ == "__main__":
    if os.path.isdir(root_image_dir):
        shutil.rmtree(root_image_dir)

    # create imagery
    for directory in image_directories:
        os.makedirs(os.path.join(root_image_dir, directory["path"]))

        # create separate bands
        for f in band_filenames:
            file_path = os.path.join(root_image_dir, directory["path"], f)
            create_imagery(path=file_path, dtype="int32", num_channels=1)

        # create stac.json
        create_stac_imagery(
            os.path.join(root_image_dir, directory["path"], "stac.json"),
            directory["bbox"],
            directory["datetime"],
        )

    # create label and corresponding stac.sjon
    if os.path.isdir(root_label_dir):
        shutil.rmtree(root_label_dir)

    for label_dir in label_directories:
        label_dir_path = os.path.join(root_label_dir, label_dir["path"])
        os.makedirs(label_dir_path)
        create_label(
            os.path.join(label_dir_path, "labels.geojson"), label_dir["num_features"]
        )
        create_stac_labels(os.path.join(label_dir_path, "stac.json"))

    # tar directories to .tar.gz and compute checksum
    for directory in [root_image_dir, root_label_dir]:
        output_filename = directory + ".tar.gz"
        make_tarfile(output_filename, directory)
        # Compute checksums
        with open(output_filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{directory}: {md5}")
