#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Regenerate the Rasteret test collection index.

Run manually to rebuild ``s2_records`` from an existing torchgeo GeoTIFF::

    python tests/data/rasteret/data.py

Rasteret's ``build_from_table(enrich_cog=True)`` parses the COG header over an
async I/O path, so it cannot run under the test suite's ``--disable-socket``.
The resulting index is committed and loaded read-only (socket-free) by the tests.
"""

from datetime import UTC, datetime

import geopandas as gpd
import rasteret
import rasterio
from shapely.geometry import box

TIF = 'tests/data/s2_100k/images/patch_0.tif'


def main() -> None:
    """Build the collection index from the Sentinel-2 patch and persist it."""
    with rasterio.open(TIF) as src:
        geometry, crs = box(*src.bounds), src.crs

    # Two dated records over the one patch: enough to exercise mosaicking,
    # time-series stacking, and spatiotemporal indexing in the tests.
    records = gpd.GeoDataFrame(
        {
            'id': ['patch_a', 'patch_b'],
            'datetime': [
                datetime(2024, 6, 1, tzinfo=UTC),
                datetime(2024, 6, 2, tzinfo=UTC),
            ],
            'assets': [{'B04': {'href': TIF}}, {'B04': {'href': TIF}}],
        },
        geometry=[geometry, geometry],
        crs=crs,
    )
    rasteret.build_from_table(
        records.to_arrow(),
        enrich_cog=True,
        band_index_map={'B04': 1},
        workspace_dir='tests/data/rasteret',
        name='s2',
    )


if __name__ == '__main__':
    main()
