import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import SouthAmericaSoybean, BoundingBox, IntersectionDataset, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSouthAmericaSoybean:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> SouthAmericaSoybean:
        monkeypatch.setattr(torchgeo.datasets.south_america_soybean, "download_url", download_url)
        transforms = nn.Identity()
        md5s = {
            2002: "8a4a9dcea54b3ec7de07657b9f2c0893",
            2021: "edff3ada13a1a9910d1fe844d28ae4f",
        }
        monkeypatch.setattr(south_america_soybean, "md5s", md5s)

        url = os.path.join("tests", "data", "south_america_soybean", "SouthAmerica_Soybean_{}.tif")
        monkeypatch.setattr(south_america_soybean, "url", url)
        

        return SouthAmericaSoybean(
            transforms=transforms,
            download=True,
            checksum=True,
            years=[2002, 2021],
        )

    def test_getitem(self, dataset: SouthAmericaSoybean) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_classes(self) -> None:
        root = os.path.join("tests", "data", "southamerica_soybean")
        classes = list(south_america_soybean.cmap.keys())[0:2]
        ds = south_america_soybean(root, years=[2021], classes=classes)
        sample = ds[ds.bounds]
        mask = sample["mask"]
        assert mask.max() < len(classes)

    def test_and(self, dataset: south_america_soybean) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: south_america_soybean) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: south_america_soybean) -> None:
        south_america_soybean(dataset.paths, download=True, years=[2021])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "south_america_soybean", "SouthAmerica_Soybean_2021.tif")
        root = str(tmp_path)

        shutil.copy(pathname, root)
        south_america_soybean(root, years=[2021])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match="SouthAmericaSoybean data product only exists for the following years:",
        ):
            south_america_soybean(str(tmp_path), years=[1996])

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            SouthAmericaSoybean(classes=[-1])

        with pytest.raises(AssertionError):
            south_america_soybean(classes=[11])

    def test_plot(self, dataset: south_america_soybean) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: south_america_soybean) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SouthAmericaSoybean(str(tmp_path))

    def test_invalid_query(self, dataset: SouthAmericaSoybean) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]