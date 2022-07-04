# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo
from torchgeo.datasets import (
    AbovegroundLiveWoodyBiomassDensity,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestAbovegroundLiveWoodyBiomassDensity:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> AbovegroundLiveWoodyBiomassDensity:

        transforms = nn.Identity()
        monkeypatch.setattr(
            torchgeo.datasets.agb_live_woody_density, "download_url", download_url
        )
        url = os.path.join(
            "tests",
            "data",
            "agb_live_woody_density",
            "Aboveground_Live_Woody_Biomass_Density.geojson",
        )
        monkeypatch.setattr(AbovegroundLiveWoodyBiomassDensity, "url", url)

        root = str(tmp_path)
        return AbovegroundLiveWoodyBiomassDensity(
            root, transforms=transforms, download=True
        )

    def test_getitem(self, dataset: AbovegroundLiveWoodyBiomassDensity) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_no_dataset(self) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in."):
            AbovegroundLiveWoodyBiomassDensity(root="/test")

    def test_already_downloaded(
        self, dataset: AbovegroundLiveWoodyBiomassDensity
    ) -> None:
        AbovegroundLiveWoodyBiomassDensity(dataset.root)

    def test_and(self, dataset: AbovegroundLiveWoodyBiomassDensity) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: AbovegroundLiveWoodyBiomassDensity) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: AbovegroundLiveWoodyBiomassDensity) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: AbovegroundLiveWoodyBiomassDensity) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()
