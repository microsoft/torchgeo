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

from torchgeo.datasets import (
    DatasetNotFoundError,
    RGBBandsMissingError,
    RwandaFieldBoundary,
)


class Collection:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join("tests", "data", "rwanda_field_boundary", "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Collection:
    return Collection()


class TestRwandaFieldBoundary:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> RwandaFieldBoundary:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.3")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch)
        monkeypatch.setattr(
            RwandaFieldBoundary, "number_of_patches_per_split", {"train": 5, "test": 5}
        )
        monkeypatch.setattr(
            RwandaFieldBoundary,
            "md5s",
            {
                "train_images": "af9395e2e49deefebb35fa65fa378ba3",
                "test_images": "d104bb82323a39e7c3b3b7dd0156f550",
                "train_labels": "6cceaf16a141cf73179253a783e7d51b",
            },
        )

        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return RwandaFieldBoundary(
            root, split, transforms=transforms, api_key="", download=True, checksum=True
        )

    def test_getitem(self, dataset: RwandaFieldBoundary) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        if dataset.split == "train":
            assert isinstance(x["mask"], torch.Tensor)
        else:
            assert "mask" not in x

    def test_len(self, dataset: RwandaFieldBoundary) -> None:
        assert len(dataset) == 5

    def test_add(self, dataset: RwandaFieldBoundary) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 10

    def test_needs_extraction(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        for fn in [
            "nasa_rwanda_field_boundary_competition_source_train.tar.gz",
            "nasa_rwanda_field_boundary_competition_source_test.tar.gz",
            "nasa_rwanda_field_boundary_competition_labels_train.tar.gz",
        ]:
            url = os.path.join("tests", "data", "rwanda_field_boundary", fn)
            shutil.copy(url, root)
        RwandaFieldBoundary(root, checksum=False)

    def test_already_downloaded(self, dataset: RwandaFieldBoundary) -> None:
        RwandaFieldBoundary(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            RwandaFieldBoundary(str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        for fn in [
            "nasa_rwanda_field_boundary_competition_source_train.tar.gz",
            "nasa_rwanda_field_boundary_competition_source_test.tar.gz",
            "nasa_rwanda_field_boundary_competition_labels_train.tar.gz",
        ]:
            with open(os.path.join(tmp_path, fn), "w") as f:
                f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            RwandaFieldBoundary(root=str(tmp_path), checksum=True)

    def test_failed_download(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.3")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch)
        monkeypatch.setattr(
            RwandaFieldBoundary,
            "md5s",
            {"train_images": "bad", "test_images": "bad", "train_labels": "bad"},
        )
        root = str(tmp_path)
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            RwandaFieldBoundary(root, "train", api_key="", download=True, checksum=True)

    def test_no_api_key(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Must provide an API key to download"):
            RwandaFieldBoundary(str(tmp_path), api_key=None, download=True)

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError, match="is an invalid band name."):
            RwandaFieldBoundary(bands=("foo", "bar"))

    def test_plot(self, dataset: RwandaFieldBoundary) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        if dataset.split == "train":
            x["prediction"] = x["mask"].clone()
            dataset.plot(x)
            plt.close()

    def test_failed_plot(self, dataset: RwandaFieldBoundary) -> None:
        single_band_dataset = RwandaFieldBoundary(root=dataset.root, bands=("B01",))
        with pytest.raises(
            RGBBandsMissingError, match="Dataset does not contain some of the RGB bands"
        ):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle="Test")
