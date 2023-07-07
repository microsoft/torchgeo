# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import AgriFieldNet


class Collection:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join("tests", "data", "agrifieldnet", "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Collection:
    return Collection()


class TestAgriFieldNet:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> AgriFieldNet:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.3")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch)
        md5 = "42fea66d90af2d854939b60e4d7fa69f"
        monkeypatch.setitem(AgriFieldNet.image_meta, "md5", md5)
        root = str(tmp_path)
        transforms = nn.Identity()
        return AgriFieldNet(
            root, transforms=transforms, download=True, api_key="", checksum=True
        )

    def test_getitem(self, dataset: AgriFieldNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["field_ids"], torch.Tensor)

    def test_len(self, dataset: AgriFieldNet) -> None:
        assert len(dataset) == 5

    def test_add(self, dataset: AgriFieldNet) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 10

    def test_already_downloaded(self, dataset: AgriFieldNet) -> None:
        AgriFieldNet(root=dataset.root, download=True, api_key="")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            AgriFieldNet(str(tmp_path))

    def test_invalid_tile(self, dataset: AgriFieldNet) -> None:
        with pytest.raises(AssertionError):
            dataset._load_label_tile("foo")

        with pytest.raises(AssertionError):
            dataset._load_image_tile("foo", ("B01", "B02"))

    def test_get_splits(self, dataset: AgriFieldNet) -> None:
        train_field_ids, test_field_ids = dataset.get_splits()
        assert isinstance(train_field_ids, list)
        assert isinstance(test_field_ids, list)
        assert len(np.unique(train_field_ids)) == 20
        assert len(np.unique(test_field_ids)) == 20
        assert 8 in train_field_ids
        assert 5 not in test_field_ids
        assert 20 in test_field_ids
        assert 25 not in train_field_ids

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            AgriFieldNet(bands=["B01", "B02"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="is an invalid band name."):
            AgriFieldNet(bands=("foo", "bar"))

    def test_plot(self, dataset: AgriFieldNet) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="Prediction")
        plt.close()

    def test_plot_rgb(self, dataset: AgriFieldNet) -> None:
        dataset = AgriFieldNet(root=dataset.root, bands=tuple(["B01"]))
        with pytest.raises(
            ValueError, match="Dataset does not contain some of the RGB bands"
        ):
            dataset.plot(dataset[0], suptitle="Single Band")
