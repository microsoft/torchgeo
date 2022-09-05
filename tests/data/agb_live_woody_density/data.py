#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os

import numpy as np
import rasterio

SIZE = 32

np.random.seed(0)


base_file = {
    "type": "FeatureCollection",
    "name": "Aboveground_Live_Woody_Biomass_Density",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "tile_id": "00N_000E",
                "download": os.path.join(
                    "tests", "data", "agb_live_woody_density", "00N_000E.tif"
                ),
                "ObjectId": 1,
                "Shape__Area": 1245542622548.8701,
                "Shape__Length": 4464169.7655813899,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [10.0, 0.0], [10.0, -10.0], [0.0, -10.0], [0.0, 0.0]]
                ],
            },
        }
    ],
}


def create_file(path: str, dtype: str, num_channels: int) -> None:
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


if __name__ == "__main__":
    base_file_name = "Aboveground_Live_Woody_Biomass_Density.geojson"
    if os.path.exists(base_file_name):
        os.remove(base_file_name)

    with open(base_file_name, "w") as f:
        json.dump(base_file, f)

    for i in base_file["features"]:
        filepath = os.path.basename(i["properties"]["download"])
        create_file(path=filepath, dtype="int32", num_channels=1)
