#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Generate small STAC GeoParquet test data: local rasters and an item table."""

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

# Four 64x64 scenes on a 2x2 grid over two days, in UTM 32N, with cloud cover.
EPSG = 32632
SIZE = 64
RES = 10.0
ITEMS = [
    ('item-0', 500000.0, 5200000.0, '2020-01-01T10:00:00Z', 5.0),
    ('item-1', 500640.0, 5200000.0, '2020-01-01T10:10:00Z', 15.0),
    ('item-2', 500000.0, 5199360.0, '2020-01-02T10:00:00Z', 25.0),
    ('item-3', 500640.0, 5199360.0, '2020-01-02T10:10:00Z', 35.0),
]


def create_stac_fixture(root: Path, asset_keys: Sequence[str] = ('B04', 'B08')) -> Path:
    """Write raster assets and ``items.parquet`` under ``root``.

    Args:
        root: Output directory for ``rasters/`` and ``items.parquet``.
        asset_keys: Asset keys to create per item.

    Returns:
        Path to the written GeoParquet item table.
    """
    (root / 'rasters').mkdir(parents=True, exist_ok=True)
    crs = CRS.from_epsg(EPSG)
    rows, geometries = [], []

    for item_index, (item_id, xmin, ymax, timestamp, cloud_cover) in enumerate(ITEMS):
        transform = from_origin(xmin, ymax, RES, RES)
        geometries.append(box(xmin, ymax - SIZE * RES, xmin + SIZE * RES, ymax))

        assets = {}
        for band_index, key in enumerate(asset_keys):
            href = root / 'rasters' / f'{item_id}_{key}.tif'
            pixels = np.full(
                (SIZE, SIZE), (item_index + 1) * (band_index + 1), 'uint16'
            )
            with rasterio.open(
                href,
                'w',
                driver='GTiff',
                height=SIZE,
                width=SIZE,
                count=1,
                dtype='uint16',
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(pixels, 1)
            assets[key] = {
                'href': str(href.relative_to(root)),
                'gsd': RES,
                'proj:transform': [transform[i] for i in range(6)],
            }

        rows.append(
            {
                'type': 'Feature',
                'stac_version': '1.1.0',
                'id': item_id,
                'collection': 'torchgeo-stac-fixture',
                'datetime': pd.Timestamp(timestamp),
                'assets': assets,
                'proj:epsg': EPSG,
                'eo:cloud_cover': cloud_cover,
            }
        )

    table = gpd.GeoDataFrame(rows, geometry=geometries, crs=crs).to_crs(4326)
    path = root / 'items.parquet'
    table.to_parquet(path, write_covering_bbox=True)
    return path


if __name__ == '__main__':
    create_stac_fixture(Path(__file__).parent)
