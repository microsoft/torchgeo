# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    GlobBiomass,
    IntersectionDataset,
    UnionDataset,
)


class TestGlobBiomass:
    @pytest.fixture
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> GlobBiomass:
        zipfile = "N00E020_agb.zip"

        shutil.copy(os.path.join("tests", "data", "globbiomass", zipfile), tmp_path)

        md5s = {zipfile: "bda282c7e5c9417397ac0463e361208b"}

        monkeypatch.setattr(GlobBiomass, "md5s", md5s)  # type: ignore[attr-defined]
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return GlobBiomass(root, transforms=transforms, checksum=True)

    def test_getitem(self, dataset: GlobBiomass) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["error_mask"], torch.Tensor)

    def test_already_extracted(self, dataset: GlobBiomass) -> None:
        GlobBiomass(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            GlobBiomass(str(tmp_path), checksum=True)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "N00E020_agb.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            GlobBiomass(root=str(tmp_path), checksum=True)

    def test_and(self, dataset: GlobBiomass) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: GlobBiomass) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: GlobBiomass) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["mask"])
        dataset.plot(x["error_mask"])

    def test_invalid_query(self, dataset: GlobBiomass) -> None:
        query = BoundingBox(100, 100, 100, 100, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
