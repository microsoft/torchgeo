# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

from rasterio import Affine
from rasterio.crs import CRS

from tests.data.cdl.data import ensure_cdl_data
from tests.data.tessera.data import ensure_tessera_data

SIZE = 64  # patch size for Tessera embeddings and CDL labels
YEAR = '2023'
affine = Affine(30.0, 0.0, 399960.0, 0.0, -30.0, 4500000.0)


def ensure_tessera_cdl_data(tessera_root: Path, cdl_root: Path | None = None) -> None:
    """Ensure Tessera and CDL test data exists with overlapping footprints."""
    tessera_root = Path(tessera_root)
    cdl_root = Path(cdl_root) if cdl_root is not None else tessera_root.parent / 'cdl'

    for split in ['train', 'val']:
        split_root = tessera_root / split
        ensure_tessera_data(
            split_root,
            size=SIZE,
            crs=CRS.from_epsg(32616),
            transform=affine,
            year=YEAR,
            subfolder='global_0.1_degree_representation',
        )

    ensure_cdl_data(cdl_root, size=SIZE, crs='epsg:32616', transform=affine)
