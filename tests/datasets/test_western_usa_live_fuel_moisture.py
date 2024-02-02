# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, WesternUSALiveFuelMoisture


class Collection:
    def download(self, output_dir: str, **kwargs: str) -> None:
        tarball_path = os.path.join(
            "tests",
            "data",
            "western_usa_live_fuel_moisture",
            "su_sar_moisture_content.tar.gz",
        )
        shutil.copy(tarball_path, output_dir)


def fetch(collection_id: str, **kwargs: str) -> Collection:
    return Collection()


class TestWesternUSALiveFuelMoisture:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> WesternUSALiveFuelMoisture:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.3")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch)
        md5 = "ecbc9269dd27c4efe7aa887960054351"
        monkeypatch.setattr(WesternUSALiveFuelMoisture, "md5", md5)
        root = str(tmp_path)
        transforms = nn.Identity()
        return WesternUSALiveFuelMoisture(
            root, transforms=transforms, download=True, api_key="", checksum=True
        )

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_getitem(self, dataset: WesternUSALiveFuelMoisture, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x["input"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: WesternUSALiveFuelMoisture) -> None:
        assert len(dataset) == 3

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            "tests",
            "data",
            "western_usa_live_fuel_moisture",
            "su_sar_moisture_content.tar.gz",
        )
        root = str(tmp_path)
        shutil.copy(pathname, root)
        WesternUSALiveFuelMoisture(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            WesternUSALiveFuelMoisture(str(tmp_path))

    def test_invalid_features(self, dataset: WesternUSALiveFuelMoisture) -> None:
        with pytest.raises(AssertionError, match="Invalid input variable name."):
            WesternUSALiveFuelMoisture(dataset.root, input_features=["foo"])
