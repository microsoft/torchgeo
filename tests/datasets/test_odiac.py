# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
import pytest
import torch
from torch.nn import Identity 
from rasterio.crs import CRS
import matplotlib.pyplot as plt

from torchgeo.datasets import (
    ODIAC,
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)

# Define root using Path for consistency
TEST_DATA_ROOT = Path("tests") / "data" / "odiac"

class TestODIAC:
    @pytest.fixture
    def dataset(self) -> ODIAC:
        root = TEST_DATA_ROOT
        transforms = Identity()
        return ODIAC(
            paths=str(root),
            version=2023,
            years=[2021, 2022],
            months=[1, 7],
            transforms=transforms,
        )

    def test_getitem(self, dataset: ODIAC) -> None:
        """Test __getitem__ returns the correct data structure."""
        # Query using the dataset's overall bounds (simpler than specific item)
        # Relying on fake data being small enough for this to work
        query_box = dataset.bounds

        # Ensure the query has non-zero spatial extent for RasterDataset logic
        query_box = BoundingBox(
            minx=query_box.minx,
            maxx=max(query_box.maxx, query_box.minx + dataset.res[0]),
            miny=query_box.miny,
            maxy=max(query_box.maxy, query_box.miny + dataset.res[1]),
            mint=query_box.mint,
            maxt=query_box.maxt
        )

        x = dataset[query_box]
        assert isinstance(x, dict)
        assert 'image' in x
        assert 'crs' in x
        assert 'bounds' in x
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['bounds'], BoundingBox)
        assert x['image'].dtype == torch.float32
        # Check shape based on fake data size (32x32)
        assert x['image'].shape == (1, 32, 32)

    def test_len(self, dataset: ODIAC) -> None:
        """Test the __len__ method returns the correct number of files."""
        assert len(dataset) == 4

    def test_and(self, dataset: ODIAC) -> None:
        """Test the intersection operator."""
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)
        assert len(ds) == len(dataset)

    def test_or(self, dataset: ODIAC) -> None:
        """Test the union operator."""
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)
        assert len(ds) == len(dataset) * 2

    def test_plot(self, dataset: ODIAC) -> None:
        """Test the plot method."""
        # Use overall bounds for plotting test
        x = dataset[dataset.bounds].copy()
        dataset.plot(x, suptitle='Test Plot')
        plt.close()

        x['prediction'] = x['image'].clone() * 0.5
        dataset.plot(x, suptitle='Test Plot with Prediction')
        plt.close()

    def test_invalid_query(self, dataset: ODIAC) -> None:
        """Test querying outside the dataset bounds."""
        query = BoundingBox(minx=200, maxx=201, miny=100, maxy=101, mint=0, maxt=1)
        with pytest.raises(IndexError, match='query: .* not found in index'):
            dataset[query]

    def test_no_data(self, tmp_path: Path) -> None:
        """Test error when no data is found and download=False."""
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ODIAC(tmp_path, download=False)

    def test_invalid_version(self, tmp_path: Path) -> None:
        """Test error with invalid version."""
        with pytest.raises(AssertionError, match='Invalid version'):
            ODIAC(tmp_path, version=1999, download=False)

    def test_invalid_year(self, tmp_path: Path) -> None:
        """Test error with invalid year for the specified version."""
        with pytest.raises(AssertionError, match='Invalid year'):
            ODIAC(tmp_path, version=2023, years=[2023], download=False)
        with pytest.raises(AssertionError, match='Invalid year'):
            ODIAC(tmp_path, version=2023, years=[1999], download=False)

    def test_invalid_month(self, tmp_path: Path) -> None:
        """Test error with invalid month."""
        with pytest.raises(AssertionError, match='Invalid month 0'):
            ODIAC(tmp_path, months=[0], download=False)
        with pytest.raises(AssertionError, match='Invalid month 13'):
            ODIAC(tmp_path, months=[13], download=False)

    def test_different_crs(self, dataset: ODIAC) -> None:
        """Test instantiating with a different CRS (attribute check only)."""
        target_crs = CRS.from_epsg(3857)
        ds_reprojected = ODIAC(
            paths=str(TEST_DATA_ROOT), # Use string path
            crs=target_crs,
            version=2023,
            years=[2021],
            months=[1]
        )
        assert ds_reprojected.crs == target_crs
        # Check that the internal _crs attribute was updated
        assert ds_reprojected._crs == target_crs
        # Check that the overall bounds changed, indicating index reprojection
        assert dataset.bounds != ds_reprojected.bounds

    def test_different_res(self, dataset: ODIAC) -> None:
        """Test instantiating with a different resolution (attribute check only)."""
        target_res_val = dataset.res[0] * 2
        target_res_tuple = (target_res_val, target_res_val)
        ds_resampled = ODIAC(
            paths=str(TEST_DATA_ROOT), # Use string path
            res=target_res_val,
            version=2023,
            years=[2021],
            months=[1]
        )
        assert ds_resampled.res == target_res_tuple
        # Check that the internal _res attribute was updated
        assert ds_resampled._res == target_res_tuple