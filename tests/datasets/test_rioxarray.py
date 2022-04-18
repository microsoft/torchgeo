import numpy as np
import pytest
import torch
import xarray as xr
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    IntersectionDataset,
    RioXarrayDataset,
    UnionDataset,
)

pytest.importorskip("rioxarray")


class TestRioXarrayDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> RioXarrayDataset:
        xr_dataarray = xr.DataArray(
            data=np.random.randn(5, 3),
            coords=dict(y=[5.6, 4.5, 3.4, 2.3, 1.2], x=[6.7, 7.8, 8.9]),
            dims=["y", "x"],
        )
        xr_dataarray.rio.set_crs(input_crs="EPSG:3857")
        return RioXarrayDataset(xr_dataarray=xr_dataarray)

    def test_getitem(self, dataset: RioXarrayDataset) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["crs"], CRS)

    def test_and(self, dataset: RioXarrayDataset) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: RioXarrayDataset) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_invalid_query(self, dataset: RioXarrayDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
