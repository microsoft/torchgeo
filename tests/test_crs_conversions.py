# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math

import pyproj
import shapely.ops

import pytest

def test_crs_with_pyproj() -> None:
    src_crs = pyproj.CRS('epsg:4326')
    dst_crs = pyproj.CRS(src_crs)

    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform

    geom = {
        "type": "Polygon",
        "coordinates": [
            [
                [-125.068359375, 45.920587344733654],
                [-116.56494140625001, 45.920587344733654],
                [-116.56494140625001, 49.095452162534826],
                [-125.068359375, 49.095452162534826],
                [-125.068359375, 45.920587344733654]
            ]
        ]
    }
    geom_transformed = shapely.ops.transform(project, shapely.geometry.shape(geom))

    bounds = geom_transformed.bounds
    expected_bounds = (
        -125.068359375,
        45.920587344733654,
        -116.56494140625001,
        49.095452162534826
    )

    for i in range(4):
        assert math.isclose(bounds[i], expected_bounds[i])