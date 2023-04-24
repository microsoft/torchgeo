# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import MapInWild


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestMapInWild:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> MapInWild:
        monkeypatch.setattr(torchgeo.datasets.mapinwild, "download_url", download_url)
        md5s = {
            "ESA_WC.zip": "376acca86ede6f5d5fd8ae99f7fa3266",
            "VIIRS.zip": "51003f6be6ec39af2193464731a60703",
            "mask.zip": "b17d85d1d7ee8b0c276d56be1f431f1c",
            "s1_part1.zip": "3ec2f3df597ee81c1e9a4dab0de3813e",
            "s1_part2.zip": "fd944ec7a4a992df0f4425278d01e350",
            "s2_temporal_subset_part1.zip": "f3db9a453097ce45d4e4bd9b0655409e",
            "s2_temporal_subset_part2.zip": "eb9c056e9f7b5862ce9cc254dd047c01",
            "s2_autumn_part1.zip": "316434f6e0e241f8c4fb2bdaf2fe4d72",
            "s2_autumn_part2.zip": "71dc31990692c2226f96caf5b19e9da3",
            "s2_spring_part1.zip": "a6f2da00db8906df41e6850a22e30e83",
            "s2_spring_part2.zip": "c21632f390c7607873b8595c80372bf8",
            "s2_summer_part1.zip": "74fdef594372efa6f3cec0f4bd77731b",
            "s2_summer_part2.zip": "3515f168ea30adff4e65b62bf2f5026b",
            "s2_winter_part1.zip": "4403f6246bbc604349654d16a46db30d",
            "s2_winter_part2.zip": "9f9c88b3755c7ec43840d5d4a81adfa2",
        }
        monkeypatch.setattr(MapInWild, "md5s", md5s)
        root = os.path.join("tests", "data", "mapinwild")
        transforms = nn.Identity()
        return MapInWild(root, download=False, transforms=transforms, checksum=True)

    def test_getitem(self, dataset: MapInWild) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 15

    def test_len(self, dataset: MapInWild) -> None:
        assert len(dataset) == 8

    def test_add(self, dataset: MapInWild) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 16

    def test_check_integrity_light(self) -> None:
        root = os.path.join("tests", "data", "mapinwild")
        ds = MapInWild(root, checksum=False)
        assert isinstance(ds, MapInWild)

    def test_plot(self, dataset: MapInWild) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="prediction")
        plt.close()
