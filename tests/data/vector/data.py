#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json

# Create an L shape:
#
# +--+
# |  |
# +--+--+
# |  |  |
# +--+--+
#
# This allows us to test queries:
#
# * within the L
# * within the dataset bounding box but with no features
# * outside the dataset bounding box

geojson = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {"label_id": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"label_id": 2},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [1.0, 0.0]]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"label_id": 3},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 1.0], [0.0, 1.0]]
                ],
            },
        },
    ],
}

with open("vector_2024.geojson", "w") as f:
    json.dump(geojson, f)
