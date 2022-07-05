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
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import TropicalCycloneWindEstimation


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        for tarball in glob.iglob(os.path.join("tests", "data", "cyclone", "*.tar.gz")):
            shutil.copy(tarball, output_dir)


def fetch(collection_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestTropicalCycloneWindEstimation:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> TropicalCycloneWindEstimation:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)
        md5s = {
            "train": {
                "source": "2b818e0a0873728dabf52c7054a0ce4c",
                "labels": "c3c2b6d02c469c5519f4add4f9132712",
            },
            "test": {
                "source": "bc07c519ddf3ce88857435ddddf98a16",
                "labels": "3ca4243eff39b87c73e05ec8db1824bf",
            },
        }
        monkeypatch.setattr(TropicalCycloneWindEstimation, "md5s", md5s)
        monkeypatch.setattr(TropicalCycloneWindEstimation, "size", 1)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return TropicalCycloneWindEstimation(
            root, split, transforms, download=True, api_key="", checksum=True
        )

    @pytest.mark.parametrize("index", [0, 1])
    def test_getitem(self, dataset: TropicalCycloneWindEstimation, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["storm_id"], str)
        assert isinstance(x["relative_time"], int)
        assert isinstance(x["ocean"], int)
        assert isinstance(x["label"], int)
        assert x["image"].shape == (dataset.size, dataset.size)

    def test_len(self, dataset: TropicalCycloneWindEstimation) -> None:
        assert len(dataset) == 5

    def test_add(self, dataset: TropicalCycloneWindEstimation) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 10

    def test_already_downloaded(self, dataset: TropicalCycloneWindEstimation) -> None:
        TropicalCycloneWindEstimation(root=dataset.root, download=True, api_key="")

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            TropicalCycloneWindEstimation(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            TropicalCycloneWindEstimation(str(tmp_path))

    def test_plot(self, dataset: TropicalCycloneWindEstimation) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = torch.tensor(sample["label"])
        dataset.plot(sample)
        plt.close()
