# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

from rasterio import Affine
from rasterio.crs import CRS

from tests.data.tessera.data import ensure_tessera_data

SIZE = 64  # patch size for Tessera embeddings, sized to fit inside tests/data/cdl
YEAR = '2023'
# Matches the origin/resolution of the CDL fixture at tests/data/cdl so the two
# datasets overlap.
affine = Affine(30.0, 0.0, 399960.0, 0.0, -30.0, 4500000.0)


def ensure_tessera_cdl_data(tessera_root: Path) -> None:
    """Ensure Tessera test data exists overlapping the tests/data/cdl fixture."""
    tessera_root = Path(tessera_root)

    for split in ['train', 'val', 'test']:
        split_root = tessera_root / split
        ensure_tessera_data(
            split_root,
            size=SIZE,
            crs=CRS.from_epsg(32616),
            transform=affine,
            year=YEAR,
            subfolder='global_0.1_degree_representation',
        )
