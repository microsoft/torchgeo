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
from pytest import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    EuroCrops,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestEuroCrops:
    @pytest.fixture(params=[None, ["1000000010"], ["1000000000"], ["2000000000"]])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> EuroCrops:
        classes = request.param
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        monkeypatch.setattr(torchgeo.datasets.eurocrops, "download_url", download_url)
        monkeypatch.setattr(
            EuroCrops, "zenodo_files", [("AA.zip", "9a45308fc32b535318562c1394fd2911")]
        )
        monkeypatch.setattr(EuroCrops, "hcat_md5", "22d61cf3b316c8babfd209ae81419d8f")
        base_url = os.path.join("tests", "data", "eurocrops") + os.sep
        monkeypatch.setattr(EuroCrops, "base_url", base_url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return EuroCrops(
            root, classes=classes, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: EuroCrops) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: EuroCrops) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EuroCrops) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: EuroCrops) -> None:
        EuroCrops(dataset.paths, download=True)

    def test_plot(self, dataset: EuroCrops) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")

    def test_plot_prediction(self, dataset: EuroCrops) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            EuroCrops(str(tmp_path))

    def test_invalid_query(self, dataset: EuroCrops) -> None:
        query = BoundingBox(200, 200, 200, 200, 2, 2)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_integrity_error(self, dataset: EuroCrops) -> None:
        dataset.zenodo_files = [("AA.zip", "invalid")]
        assert not dataset._check_integrity()
