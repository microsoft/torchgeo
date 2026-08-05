# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import pytest
import rasterio
import shapely
import shapely.wkt
import torch
from _pytest.fixtures import SubRequest
from geopandas import GeoSeries
from geopandas.testing import assert_geoseries_equal
from torch import nn

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    Sentinel1,
    Sentinel2,
    UnionDataset,
)


class TestSentinel1:
    @pytest.fixture(
        params=[
            # Only horizontal or vertical receive
            ['HH'],
            ['HV'],
            ['VV'],
            ['VH'],
            # Both horizontal and vertical receive
            ['HH', 'HV'],
            ['HV', 'HH'],
            ['VV', 'VH'],
            ['VH', 'VV'],
        ]
    )
    def dataset(self, request: SubRequest) -> Sentinel1:
        root = os.path.join('tests', 'data', 'sentinel1')
        bands = request.param
        transforms = nn.Identity()
        return Sentinel1(root, bands=bands, transforms=transforms)

    def test_getitem(self, dataset: Sentinel1) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Sentinel1) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: Sentinel1) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel1) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Sentinel1(tmp_path)

    def test_empty_bands(self) -> None:
        with pytest.raises(AssertionError, match="'bands' cannot be an empty list"):
            Sentinel1(bands=[])

    @pytest.mark.parametrize('bands', [['HH', 'HH'], ['HH', 'HV', 'HH']])
    def test_duplicate_bands(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="'bands' contains duplicate bands"):
            Sentinel1(bands=bands)

    @pytest.mark.parametrize('bands', [['HH_HV'], ['HH', 'HV', 'HH_HV']])
    def test_invalid_bands(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="invalid band 'HH_HV'"):
            Sentinel1(bands=bands)

    @pytest.mark.parametrize(
        'bands', [['HH', 'VV'], ['HH', 'VH'], ['VV', 'HV'], ['HH', 'HV', 'VV', 'VH']]
    )
    def test_dual_transmit(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="'bands' cannot contain both "):
            Sentinel1(bands=bands)

    def test_invalid_index(self, dataset: Sentinel1) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[-1:-1, -1:-1, pd.Timestamp.min : pd.Timestamp.min]


class TestSentinel2:
    @pytest.fixture
    def dataset(self) -> Sentinel2:
        root = os.path.join('tests', 'data', 'sentinel2')
        res = (10.0, 10.0)
        transforms = nn.Identity()
        bands = [
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B09',
            'B11',
            'B12',
        ]
        return Sentinel2(root, res=res, transforms=transforms, bands=bands)

    def test_getitem(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Sentinel2) -> None:
        assert len(dataset) == 4

    def test_and(self, dataset: Sentinel2) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel2) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Sentinel2(tmp_path)

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_wrong_bands(self, dataset: Sentinel2) -> None:
        bands = ['B02']
        ds = Sentinel2(dataset.paths, res=dataset.res, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(x)

    def test_invalid_index(self, dataset: Sentinel2) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_float_res(self, dataset: Sentinel2) -> None:
        Sentinel2(dataset.paths, res=10.0, bands=dataset.bands)

    @pytest.mark.parametrize('crs', [None, pyproj.CRS('EPSG:3857')])
    def test_true_footprint_from_metadata(self, crs: pyproj.CRS | None) -> None:
        root = os.path.join('tests', 'data', 'sentinel2')
        ds = Sentinel2(root, res=(10.0, 10.0), crs=crs, bands=['B02'])

        def read_footprint_wkt(filepath: str) -> str:
            metadata_path = filepath.split('GRANULE')[0] + 'MTD_MSIL1C.xml'
            with rasterio.open(metadata_path) as src:
                return src.tags()['FOOTPRINT']

        expected = GeoSeries(
            ds.index['filepath'].map(read_footprint_wkt).map(shapely.wkt.loads),
            crs='EPSG:4326',
        ).to_crs(ds.crs)
        assert_geoseries_equal(ds.index.geometry, expected)

    def test_footprint_falls_back_to_bbox(
        self, dataset: Sentinel2, tmp_path: Path
    ) -> None:
        filepath = next(iter(dataset.files))
        # Move this raster to a directory where it does not find metadata file
        link = tmp_path / Path(filepath).name
        link.symlink_to(Path(filepath).resolve())

        with rasterio.open(link) as src:
            result = dataset.footprint_from_datasource(src)
            assert result.equals_exact(shapely.box(*src.bounds), tolerance=1e-9)
