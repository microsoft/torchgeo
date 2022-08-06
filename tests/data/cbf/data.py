#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import os
import shutil


def create_geojson():
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            }
        ],
    }
    return geojson


if __name__ == "__main__":
    filename = "Alberta.zip"
    geojson = create_geojson()

    with open(filename.replace(".zip", ".geojson"), "w") as f:
        json.dump(geojson, f)

    # compress single file directly with no directory
    shutil.make_archive(
        filename.replace(".zip", ""),
        "zip",
        os.getcwd(),
        filename.replace(".zip", ".geojson"),
    )

    # Compute checksums
    with open(filename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{filename}: {md5}")
