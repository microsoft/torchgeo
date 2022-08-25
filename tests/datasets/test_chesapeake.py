# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import (
    BoundingBox,
    Chesapeake13,
    ChesapeakeCVPR,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestChesapeake13:
    pytest.importorskip("zipfile_deflate64")

    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> Chesapeake13:
        monkeypatch.setattr(torchgeo.datasets.chesapeake, "download_url", download_url)
        md5 = "fe35a615b8e749b21270472aa98bb42c"
        monkeypatch.setattr(Chesapeake13, "md5", md5)
        url = os.path.join(
            "tests", "data", "chesapeake", "BAYWIDE", "Baywide_13Class_20132014.zip"
        )
        monkeypatch.setattr(Chesapeake13, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return Chesapeake13(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: Chesapeake13) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: Chesapeake13) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Chesapeake13) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: Chesapeake13) -> None:
        Chesapeake13(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        url = os.path.join(
            "tests", "data", "chesapeake", "BAYWIDE", "Baywide_13Class_20132014.zip"
        )
        root = str(tmp_path)
        shutil.copy(url, root)
        Chesapeake13(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            Chesapeake13(str(tmp_path), checksum=True)

    def test_plot(self, dataset: Chesapeake13) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: Chesapeake13) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_url(self) -> None:
        ds = Chesapeake13(os.path.join("tests", "data", "chesapeake", "BAYWIDE"))
        assert "cicwebresources.blob.core.windows.net" in ds.url

    def test_invalid_query(self, dataset: Chesapeake13) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]


class TestChesapeakeCVPR:
    @pytest.fixture(
        params=[
            ("naip-new", "naip-old", "nlcd"),
            ("landsat-leaf-on", "landsat-leaf-off", "lc"),
            ("naip-new", "landsat-leaf-on", "lc", "nlcd", "buildings"),
            ("naip-new", "prior_from_cooccurrences_101_31_no_osm_no_buildings"),
        ]
    )
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> ChesapeakeCVPR:
        monkeypatch.setattr(torchgeo.datasets.chesapeake, "download_url", download_url)
        monkeypatch.setattr(
            ChesapeakeCVPR,
            "md5s",
            {
                "base": "882d18b1f15ea4498bf54e674aecd5d4",
                "prior_extension": "677446c486f3145787938b14ee3da13f",
            },
        )
        monkeypatch.setattr(
            ChesapeakeCVPR,
            "urls",
            {
                "base": os.path.join(
                    "tests",
                    "data",
                    "chesapeake",
                    "cvpr",
                    "cvpr_chesapeake_landcover.zip",
                ),
                "prior_extension": os.path.join(
                    "tests",
                    "data",
                    "chesapeake",
                    "cvpr",
                    "cvpr_chesapeake_landcover_prior_extension.zip",
                ),
            },
        )
        monkeypatch.setattr(
            ChesapeakeCVPR,
            "files",
            ["de_1m_2013_extended-debuffered-test_tiles", "spatial_index.geojson"],
        )
        root = str(tmp_path)
        transforms = nn.Identity()
        return ChesapeakeCVPR(
            root,
            splits=["de-test"],
            layers=request.param,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: ChesapeakeCVPR) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: ChesapeakeCVPR) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: ChesapeakeCVPR) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: ChesapeakeCVPR) -> None:
        ChesapeakeCVPR(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        shutil.copy(
            os.path.join(
                "tests", "data", "chesapeake", "cvpr", "cvpr_chesapeake_landcover.zip"
            ),
            root,
        )
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "chesapeake",
                "cvpr",
                "cvpr_chesapeake_landcover_prior_extension.zip",
            ),
            root,
        )
        ChesapeakeCVPR(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            ChesapeakeCVPR(str(tmp_path), checksum=True)

    def test_out_of_bounds_query(self, dataset: ChesapeakeCVPR) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_multiple_hits_query(self, dataset: ChesapeakeCVPR) -> None:
        ds = ChesapeakeCVPR(
            root=dataset.root, splits=["de-train", "de-test"], layers=dataset.layers
        )
        with pytest.raises(
            IndexError, match="query: .* spans multiple tiles which is not valid"
        ):
            ds[dataset.bounds]
