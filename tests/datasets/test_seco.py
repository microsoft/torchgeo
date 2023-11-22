# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import (
    DatasetNotFoundError,
    RGBBandsMissingError,
    SeasonalContrastS2,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSeasonalContrastS2:
    @pytest.fixture(
        params=zip(
            ["100k", "1m"],
            [1, 2],
            [SeasonalContrastS2.rgb_bands, SeasonalContrastS2.all_bands],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SeasonalContrastS2:
        monkeypatch.setattr(torchgeo.datasets.seco, "download_url", download_url)
        monkeypatch.setitem(
            SeasonalContrastS2.metadata["100k"],
            "url",
            os.path.join("tests", "data", "seco", "seco_100k.zip"),
        )
        monkeypatch.setitem(
            SeasonalContrastS2.metadata["100k"],
            "md5",
            "6f527567f066562af2c78093114599f9",
        )
        monkeypatch.setitem(
            SeasonalContrastS2.metadata["1m"],
            "url",
            os.path.join("tests", "data", "seco", "seco_1m.zip"),
        )
        monkeypatch.setitem(
            SeasonalContrastS2.metadata["1m"], "md5", "3bb3fcf90f5de7d5781ce0cb85fd20af"
        )
        root = str(tmp_path)
        version, seasons, bands = request.param
        transforms = nn.Identity()
        return SeasonalContrastS2(
            root, version, seasons, bands, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: SeasonalContrastS2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].size(0) == dataset.seasons * len(dataset.bands)

    def test_len(self, dataset: SeasonalContrastS2) -> None:
        if dataset.version == "100k":
            assert len(dataset) == 10**5 // 5
        else:
            assert len(dataset) == 10**6 // 5

    def test_add(self, dataset: SeasonalContrastS2) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        if dataset.version == "100k":
            assert len(ds) == 2 * 10**5 // 5
        else:
            assert len(ds) == 2 * 10**6 // 5

    def test_already_extracted(self, dataset: SeasonalContrastS2) -> None:
        SeasonalContrastS2(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "seco", "*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        SeasonalContrastS2(root)

    def test_invalid_version(self) -> None:
        with pytest.raises(AssertionError):
            SeasonalContrastS2(version="foo")

    def test_invalid_band(self) -> None:
        with pytest.raises(AssertionError):
            SeasonalContrastS2(bands=["A1steaksauce"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            SeasonalContrastS2(str(tmp_path))

    def test_plot(self, dataset: SeasonalContrastS2) -> None:
        x = dataset[0]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        with pytest.raises(ValueError, match="doesn't support plotting"):
            x["prediction"] = torch.tensor(1)
            dataset.plot(x)

    def test_no_rgb_plot(self) -> None:
        with pytest.raises(
            RGBBandsMissingError, match="Dataset does not contain some of the RGB bands"
        ):
            root = os.path.join("tests", "data", "seco")
            dataset = SeasonalContrastS2(root, bands=["B1"])
            x = dataset[0]
            dataset.plot(x, suptitle="Test")
