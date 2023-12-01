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

import torchgeo
from torchgeo.datasets import DatasetNotFoundError, DigitalTyphoonAnalysis


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestTropicalCyclone:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DigitalTyphoonAnalysis:
        monkeypatch.setattr(
            torchgeo.datasets.digital_typhoon, "download_url", download_url
        )

        url = os.path.join("tests", "data", "digital_typhoon", "WP.tar.gz{0}")
        monkeypatch.setattr(DigitalTyphoonAnalysis, "url", url)

        md5sums = {"": "40355bf0d6112d84943de4d0ec517191"}
        monkeypatch.setattr(DigitalTyphoonAnalysis, "md5sums", md5sums)
        root = str(tmp_path)

        transforms = nn.Identity()
        return DigitalTyphoonAnalysis(
            root=root, transforms=transforms, download=True, checksum=True
        )

    @pytest.mark.parametrize("index", [0, 1])
    def test_getitem(self, dataset: DigitalTyphoonAnalysis, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_already_downloaded(self, dataset: DigitalTyphoonAnalysis) -> None:
        DigitalTyphoonAnalysis(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            DigitalTyphoonAnalysis(root=str(tmp_path))

    def test_plot(self, dataset: DigitalTyphoonAnalysis) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["label"]
        dataset.plot(sample)
        plt.close()
