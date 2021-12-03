# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

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
    ChesapeakeCVPRDataModule,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestChesapeake13:
    @pytest.fixture
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> Chesapeake13:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "9557b609e614a1f79dec6eb1bb3f3a06"
        monkeypatch.setattr(Chesapeake13, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join(
            "tests", "data", "chesapeake", "BAYWIDE", "Baywide_13Class_20132014.zip"
        )
        monkeypatch.setattr(Chesapeake13, "url", url)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
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

    def test_already_downloaded(self, dataset: Chesapeake13) -> None:
        Chesapeake13(root=dataset.root, download=True)

    def test_plot(self, dataset: Chesapeake13) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["mask"])

    def test_url(self) -> None:
        ds = Chesapeake13(os.path.join("tests", "data", "chesapeake", "BAYWIDE"))
        assert "cicwebresources.blob.core.windows.net" in ds.url

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            Chesapeake13(str(tmp_path))

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
        ]
    )
    def dataset(
        self,
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> ChesapeakeCVPR:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.chesapeake, "download_url", download_url
        )
        md5 = "882d18b1f15ea4498bf54e674aecd5d4"
        monkeypatch.setattr(ChesapeakeCVPR, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join(
            "tests", "data", "chesapeake", "cvpr", "cvpr_chesapeake_landcover.zip"
        )
        monkeypatch.setattr(ChesapeakeCVPR, "url", url)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            ChesapeakeCVPR,
            "files",
            ["de_1m_2013_extended-debuffered-test_tiles", "spatial_index.geojson"],
        )
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
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
        url = os.path.join(
            "tests", "data", "chesapeake", "cvpr", "cvpr_chesapeake_landcover.zip"
        )
        root = str(tmp_path)
        shutil.copy(url, root)
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


class TestChesapeakeCVPRDataModule:
    @pytest.fixture(scope="class", params=[5, 7])
    def datamodule(self, request: SubRequest) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patch_size=32,
            patches_per_tile=2,
            batch_size=2,
            num_workers=0,
            class_set=request.param,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.test_dataloader()))

    def test_nodata_check(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        nodata_check = datamodule.nodata_check(4)
        sample = {
            "image": torch.ones(1, 2, 2),  # type: ignore[attr-defined]
            "mask": torch.ones(2, 2),  # type: ignore[attr-defined]
        }
        out = nodata_check(sample)
        assert torch.equal(  # type: ignore[attr-defined]
            out["image"], torch.zeros(1, 4, 4)  # type: ignore[attr-defined]
        )
        assert torch.equal(  # type: ignore[attr-defined]
            out["mask"], torch.zeros(4, 4)  # type: ignore[attr-defined]
        )
