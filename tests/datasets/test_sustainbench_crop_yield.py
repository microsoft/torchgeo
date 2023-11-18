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

import torchgeo.datasets.utils
from torchgeo.datasets import DatasetNotFoundError, SustainBenchCropYield


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSustainBenchCropYield:
    @pytest.fixture(params=["train", "dev", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SustainBenchCropYield:
        monkeypatch.setattr(
            torchgeo.datasets.sustainbench_crop_yield, "download_url", download_url
        )

        md5 = "7a5591794e14dd73d2b747cd2244acbc"
        monkeypatch.setattr(SustainBenchCropYield, "md5", md5)
        url = os.path.join("tests", "data", "sustainbench_crop_yield", "soybeans.zip")
        monkeypatch.setattr(SustainBenchCropYield, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        split = request.param
        countries = ["argentina", "brazil", "usa"]
        transforms = nn.Identity()
        return SustainBenchCropYield(
            root, split, countries, transforms, download=True, checksum=True
        )

    def test_already_extracted(self, dataset: SustainBenchCropYield) -> None:
        SustainBenchCropYield(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            "tests", "data", "sustainbench_crop_yield", "soybeans.zip"
        )
        root = str(tmp_path)
        shutil.copy(pathname, root)
        SustainBenchCropYield(root)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_getitem(self, dataset: SustainBenchCropYield, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert isinstance(x["year"], torch.Tensor)
        assert isinstance(x["ndvi"], torch.Tensor)
        assert x["image"].shape == (9, 32, 32)

    def test_len(self, dataset: SustainBenchCropYield) -> None:
        assert len(dataset) == len(dataset.countries) * 3

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SustainBenchCropYield(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            SustainBenchCropYield(str(tmp_path))

    def test_plot(self, dataset: SustainBenchCropYield) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["label"]
        dataset.plot(sample)
        plt.close()
