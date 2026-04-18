# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
import shapely
import torch
from pyproj import CRS
from rasterio.transform import from_bounds

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    STACDataset,
    Sentinel2STAC,
    UnionDataset,
)

# Small UTM-style frame so we can use meter-based res and bounds.
EPSG = 32633
TILE_W, TILE_H = 32, 32
ORIGIN_X, ORIGIN_Y = 500000.0, 4000000.0
PIXEL = 10.0


def _write_tif(path: Path, value: int) -> tuple[float, float, float, float]:
    """Write a constant-value GeoTIFF and return its bounds in EPSG:32633."""
    minx = ORIGIN_X
    maxy = ORIGIN_Y
    maxx = minx + TILE_W * PIXEL
    miny = maxy - TILE_H * PIXEL
    transform = from_bounds(minx, miny, maxx, maxy, TILE_W, TILE_H)
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'count': 1,
        'width': TILE_W,
        'height': TILE_H,
        'crs': CRS.from_epsg(EPSG),
        'transform': transform,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(np.full((TILE_H, TILE_W), value, dtype=np.uint16), 1)
    return (minx, miny, maxx, maxy)


def _build_parquet(
    tmp_path: Path, *, use_assets_struct: bool = False
) -> tuple[Path, tuple[float, float, float, float]]:
    """Create one item's worth of B02/B03/B04 GeoTIFFs + a STAC GeoParquet."""
    bounds = (0.0, 0.0, 0.0, 0.0)
    hrefs: dict[str, str] = {}
    for i, band in enumerate(['B02', 'B03', 'B04']):
        tif = tmp_path / f'item1_{band}.tif'
        bounds = _write_tif(tif, value=100 + i)
        hrefs[band] = str(tif)

    geom = shapely.box(*bounds)

    if use_assets_struct:
        row = {
            'geometry': [geom],
            'datetime': ['2024-06-15T00:00:00Z'],
            'assets': [{b: {'href': h, 'type': 'image/tiff'} for b, h in hrefs.items()}],
        }
    else:
        row = {
            'geometry': [geom],
            'datetime': ['2024-06-15T00:00:00Z'],
            **{b: [h] for b, h in hrefs.items()},
        }

    df = gpd.GeoDataFrame(row, geometry='geometry', crs=f'EPSG:{EPSG}')
    parquet = tmp_path / 'items.parquet'
    df.to_parquet(parquet)
    return parquet, bounds


class TestSTACDataset:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> Sentinel2STAC:
        parquet, _ = _build_parquet(tmp_path)
        return Sentinel2STAC(paths=parquet, bands=('B04', 'B03', 'B02'))

    def test_len(self, dataset: Sentinel2STAC) -> None:
        assert len(dataset) == 1

    def test_getitem(self, dataset: Sentinel2STAC) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        # 3 bands, mosaic mode → [3, H, W]
        assert x['image'].ndim == 3
        assert x['image'].shape[0] == 3

    def test_assets_struct_schema(self, tmp_path: Path) -> None:
        parquet, _ = _build_parquet(tmp_path, use_assets_struct=True)
        ds = Sentinel2STAC(paths=parquet, bands=('B04', 'B03', 'B02'))
        x = ds[ds.bounds]
        assert x['image'].shape[0] == 3

    def test_asset_columns_remap(self, tmp_path: Path) -> None:
        parquet, _ = _build_parquet(tmp_path)
        # Re-read and rename a column to simulate non-standard schema.
        df = gpd.read_parquet(parquet)
        df = df.rename(columns={'B02': 'blue'})
        df.to_parquet(parquet)
        ds = Sentinel2STAC(
            paths=parquet,
            bands=('B04', 'B03', 'B02'),
            asset_columns={'B02': 'blue'},
        )
        assert len(ds) == 1

    def test_missing_asset_raises(self, tmp_path: Path) -> None:
        parquet, _ = _build_parquet(tmp_path)
        with pytest.raises(KeyError, match="asset 'B11' not found"):
            Sentinel2STAC(paths=parquet, bands=('B11',))

    def test_invalid_band_assertion(self, tmp_path: Path) -> None:
        parquet, _ = _build_parquet(tmp_path)
        with pytest.raises(AssertionError, match='not in Sentinel2STAC.all_bands'):
            Sentinel2STAC(paths=parquet, bands=('B99',))

    def test_dataset_not_found_empty_parquet(self, tmp_path: Path) -> None:
        # An empty parquet should raise DatasetNotFoundError.
        empty = gpd.GeoDataFrame(
            {'geometry': [], 'datetime': [], 'B02': [], 'B03': [], 'B04': []},
            geometry='geometry',
            crs='EPSG:32633',
        )
        path = tmp_path / 'empty.parquet'
        empty.to_parquet(path)
        with pytest.raises(DatasetNotFoundError):
            Sentinel2STAC(paths=path, bands=('B04', 'B03', 'B02'))

    def test_index_out_of_bounds(self, dataset: Sentinel2STAC) -> None:
        # Slice well outside the item's footprint.
        far = (slice(1e9, 1e9 + 1, 10.0), slice(1e9, 1e9 + 1, 10.0))
        with pytest.raises(IndexError, match='not found in dataset'):
            dataset[far]

    def test_plot(self, dataset: Sentinel2STAC) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='test')
        plt.close()

    def test_plot_missing_rgb(self, tmp_path: Path) -> None:
        parquet, _ = _build_parquet(tmp_path)
        ds = Sentinel2STAC(paths=parquet, bands=('B04',))
        x = ds[ds.bounds]
        with pytest.raises(RGBBandsMissingError):
            ds.plot(x)

    def test_intersection(self, dataset: Sentinel2STAC) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_union(self, dataset: Sentinel2STAC) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_subclass_uses_base(self) -> None:
        assert issubclass(Sentinel2STAC, STACDataset)
