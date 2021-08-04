import os
from pathlib import Path

import pytest
import torch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, Sentinel2, ZipDataset
from torchgeo.transforms import Identity


class TestSentinel2:
    @pytest.fixture
    def dataset(self) -> Sentinel2:
        root = os.path.join("tests", "data", "sentinel2")
        bands = ["B04", "B03", "B02"]
        transforms = Identity()
        return Sentinel2(root, bands=bands, transforms=transforms)

    def test_getitem(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_add(self, dataset: Sentinel2) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No Sentinel data was found in "):
            Sentinel2(str(tmp_path))

    def test_invalid_query(self, dataset: Sentinel2) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* is not within bounds of the index:"
        ):
            dataset[query]
