# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def create_dummy_data(root: str, tiles: list[tuple[int, int, int]]) -> None:
    """Create dummy GeoTIFFs for OpenAerialMap tests.

    Args:
        root: Root directory to save files
        tiles: List of (x, y, z) tuples simulating mercantile tiles
    """
    os.makedirs(root, exist_ok=True)

    # Using roughly the area around Banepa, Nepal
    minx, miny, maxx, maxy = 85.516, 27.631, 85.523, 27.637

    for x, y, z in tiles:
        path = os.path.join(root, f'OAM-{x}-{y}-{z}.tif')

        # Create a 3-channel dummy image (RGB)
        height, width = 256, 256
        data = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)

        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype='uint8',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data)


if __name__ == '__main__':
    dummy_tiles = [(372608, 213968, 19), (372609, 213968, 19)]
    create_dummy_data('tests/data/openaerialmap', dummy_tiles)
