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
        md5 = "aa39edc40b37d2deab4115d8c2ffeced"
        monkeypatch.setitem(AgriFieldNet.image_meta, "md5", md5)
        # root = str(tmp_path)
        root = os.path.join("tests", "data", "agrifieldnet", "ref_agrifieldnet_competition_v1.tar.tar.gz")
        transforms = nn.Identity()
        return AgriFieldNet(
            root,
            transforms,
            download=True,
            api_key="",
            checksum=True,
        )

    def test_getitem(self, dataset: AgriFieldNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        # assert isinstance(x["x"], torch.Tensor)
        # assert isinstance(x["y"], torch.Tensor)

    def test_len(self, dataset: AgriFieldNet) -> None:
        assert len(dataset) == 3

    def test_add(self, dataset: AgriFieldNet) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 6

    def test_already_downloaded(self, dataset: AgriFieldNet) -> None:
        AgriFieldNet(root=dataset.root, download=True, api_key="")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            AgriFieldNet(str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(
            os.path.join(tmp_path, "ref_agrifieldnet_competition_v1.tar.gz"), "w"
        ) as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            AgriFieldNet(root=str(tmp_path), checksum=True)

    def test_invalid_split(self, dataset: AgriFieldNet) -> None:
        train_field_ids, test_field_ids = dataset.get_splits()
        assert isinstance(train_field_ids, list)
        assert isinstance(test_field_ids, list)
        assert len(train_field_ids) == 18
        assert len(test_field_ids) == 9
        # assert 336 in train_field_ids
        # assert 336 not in test_field_ids
        # assert 4793 in test_field_ids
        # assert 4793 not in train_field_ids

    # def test_invalid_split(self) -> None:
    #     with pytest.raises(AssertionError):
    #         AgriFieldNet(split="foo")

    def test_plot(self, dataset: AgriFieldNet) -> None:
        # x = dataset[0].copy()
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, show_titles=False)
        plt.close()

    def test_plot_rgb(self, dataset: AgriFieldNet) -> None:
        dataset = AgriFieldNet(root=dataset.root, bands=tuple(["B01"]))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle="Single Band")
