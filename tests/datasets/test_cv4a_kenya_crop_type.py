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
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import CV4AKenyaCropType


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_kenya_02", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestCV4AKenyaCropType:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> CV4AKenyaCropType:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)
        source_md5 = "7f4dcb3f33743dddd73f453176308bfb"
        labels_md5 = "95fc59f1d94a85ec00931d4d1280bec9"
        monkeypatch.setitem(CV4AKenyaCropType.image_meta, "md5", source_md5)
        monkeypatch.setitem(CV4AKenyaCropType.target_meta, "md5", labels_md5)
        monkeypatch.setattr(
            CV4AKenyaCropType, "tile_names", ["ref_african_crops_kenya_02_tile_00"]
        )
        monkeypatch.setattr(CV4AKenyaCropType, "dates", ["20190606"])
        root = str(tmp_path)
        transforms = nn.Identity()
        return CV4AKenyaCropType(
            root,
            transforms=transforms,
            download=True,
            api_key="",
            checksum=True,
            verbose=True,
        )

    def test_getitem(self, dataset: CV4AKenyaCropType) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["x"], torch.Tensor)
        assert isinstance(x["y"], torch.Tensor)

    def test_len(self, dataset: CV4AKenyaCropType) -> None:
        assert len(dataset) == 345

    def test_add(self, dataset: CV4AKenyaCropType) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 690

    def test_get_splits(self, dataset: CV4AKenyaCropType) -> None:
        train_field_ids, test_field_ids = dataset.get_splits()
        assert isinstance(train_field_ids, list)
        assert isinstance(test_field_ids, list)
        assert len(train_field_ids) == 18
        assert len(test_field_ids) == 9
        assert 336 in train_field_ids
        assert 336 not in test_field_ids
        assert 4793 in test_field_ids
        assert 4793 not in train_field_ids

    def test_already_downloaded(self, dataset: CV4AKenyaCropType) -> None:
        CV4AKenyaCropType(root=dataset.root, download=True, api_key="")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CV4AKenyaCropType(str(tmp_path))

    def test_invalid_tile(self, dataset: CV4AKenyaCropType) -> None:
        with pytest.raises(AssertionError):
            dataset._load_label_tile("foo")

        with pytest.raises(AssertionError):
            dataset._load_all_image_tiles("foo", ("B01", "B02"))

        with pytest.raises(AssertionError):
            dataset._load_single_image_tile("foo", "20190606", ("B01", "B02"))

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            CV4AKenyaCropType(bands=["B01", "B02"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="is an invalid band name."):
            CV4AKenyaCropType(bands=("foo", "bar"))

    def test_plot(self, dataset: CV4AKenyaCropType) -> None:
        dataset.plot(dataset[0], time_step=0, suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, time_step=0, suptitle="Pred")
        plt.close()

    def test_plot_rgb(self, dataset: CV4AKenyaCropType) -> None:
        dataset = CV4AKenyaCropType(root=dataset.root, bands=tuple(["B01"]))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], time_step=0, suptitle="Single Band")
