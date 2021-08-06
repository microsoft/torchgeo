import os
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
from pytest import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, Landsat8, ZipDataset
from torchgeo.transforms import Identity


class TestLandsat8:
    @pytest.fixture
    def dataset(self, monkeypatch: Generator[MonkeyPatch, None, None]) -> Landsat8:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        root = os.path.join("tests", "data", "landsat8")
        bands = ["B4", "B3", "B2"]
        transforms = Identity()
        return Landsat8(root, bands=bands, transforms=transforms)

    def test_getitem(self, dataset: Landsat8) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_add(self, dataset: Landsat8) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_plot(self, dataset: Landsat8) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["image"])

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No Landsat8 data was found in "):
            Landsat8(str(tmp_path))

    def test_invalid_query(self, dataset: Landsat8) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
