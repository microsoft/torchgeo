#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

WIDTH = 36
HEIGHT = 36
BANDS = 13
PIXEL_SIZE = 30.0  # meters per pixel
np.random.seed(0)

# EPSG 32602 / UTM zone 2N using PROJ string (bypass proj.db issues)
CRS_32602 = CRS.from_string('+proj=utm +zone=2 +north +datum=WGS84 +units=m +no_defs')


def write_mock_geotiff(
    path: str,
    origin_x: float,
    origin_y: float,
    width: int = WIDTH,
    height: int = HEIGHT,
    bands: int = BANDS,
    crs: CRS = CRS_32602,
) -> None:
    """
    Create a mock GeoTIFF with random UInt16 data.

    Args:
        path: Output file path (parent directories created automatically)
        origin_x: Upper-left X coordinate
        origin_y: Upper-left Y coordinate
        width: Number of pixels in X direction
        height: Number of pixels in Y direction
        bands: Number of raster bands
        crs: Coordinate Reference System
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define affine transform (origin top-left, Y decreasing)
    transform = from_bounds(
        origin_x,
        origin_y - height * PIXEL_SIZE,  # minx, miny
        origin_x + width * PIXEL_SIZE,
        origin_y,  # maxx, maxy
        width,
        height,
    )

    # Raster profile
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'count': bands,
        'width': width,
        'height': height,
        'crs': crs,
        'transform': transform,
        'compress': 'deflate',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'interleave': 'band',
        'nodata': None,
    }

    # Generate random raster data
    data = np.random.randint(0, 64000, size=(bands, height, width), dtype=np.uint16)
    # Band 13: low-range discrete QA/class band
    data[12] = np.random.randint(0, 10, size=(height, width), dtype=np.uint16)

    # Write GeoTIFF
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)

    print(f'Written: {path}')


# 2024 image (02VMN)
write_mock_geotiff(
    path=os.path.join(
        'tests',
        'data',
        'esd',
        'SDC30_EBD_V001',
        '2024',
        'SDC30_EBD_V001_02VMN_2024_mock.tif',
    ),
    origin_x=399945.0,
    origin_y=6800055.0,
)
