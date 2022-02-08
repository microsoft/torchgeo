# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    CMS_Global_Mangrove_Canopy,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestCMS_Global_Mangrove_Canopy:
    @pytest.fixture
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> CMS_Global_Mangrove_Canopy:
        zipfile = "CMS_Global_Map_Mangrove_Canopy_1665.zip"
        monkeypatch.setattr(  # type: ignore[attr-defined]
            CMS_Global_Mangrove_Canopy, "zipfile", zipfile
        )

        md5 = "c41917ef6bb76264f5b0e01de20c728d"
        monkeypatch.setattr(  # type: ignore[attr-defined]
            CMS_Global_Mangrove_Canopy, "md5", md5
        )

        root = os.path.join("tests", "data", "cms_mangrove_canopy")
        transforms = nn.Identity()  # type: ignore[attr-defined]
        country = "Angola"

        return CMS_Global_Mangrove_Canopy(
            root, country=country, transforms=transforms, checksum=True
        )

    def test_getitem(self, dataset: CMS_Global_Mangrove_Canopy) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_integrity(self) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CMS_Global_Mangrove_Canopy(root="/test")

    def test_invalid_country(self) -> None:
        with pytest.raises(AssertionError):
            CMS_Global_Mangrove_Canopy(country="fakeCountry")

    def test_invalid_measurement(self) -> None:
        with pytest.raises(AssertionError):
            CMS_Global_Mangrove_Canopy(measurement="wrongMeasurement")

    def test_and(self, dataset: CMS_Global_Mangrove_Canopy) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: CMS_Global_Mangrove_Canopy) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: CMS_Global_Mangrove_Canopy) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["mask"])

    def test_invalid_query(self, dataset: CMS_Global_Mangrove_Canopy) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
