# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import (
    NAIP,
    BoundingBox,
    IntersectionDataset,
    NAIPChesapeakeDataModule,
    UnionDataset,
)


class TestNAIP:
    @pytest.fixture
    def dataset(self, monkeypatch: Generator[MonkeyPatch, None, None]) -> NAIP:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        root = os.path.join("tests", "data", "naip")
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return NAIP(root, transforms=transforms)

    def test_getitem(self, dataset: NAIP) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_and(self, dataset: NAIP) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NAIP) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: NAIP) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["image"])

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No NAIP data was found in "):
            NAIP(str(tmp_path))

    def test_invalid_query(self, dataset: NAIP) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]


class TestNAIPChesapeakeDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> NAIPChesapeakeDataModule:
        dm = NAIPChesapeakeDataModule(
            os.path.join("tests", "data", "naip"),
            os.path.join("tests", "data", "chesapeake", "BAYWIDE"),
            batch_size=2,
            num_workers=0,
        )
        dm.patch_size = 32
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: NAIPChesapeakeDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: NAIPChesapeakeDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: NAIPChesapeakeDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
