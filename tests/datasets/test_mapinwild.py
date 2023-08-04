# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import MapInWild


class TestMapInWild:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> MapInWild:
        md5s = {
            "ESA_WC.zip": "b9adc4633edb872f8b5cd0f9eb7d2b14",
            "VIIRS.zip": "3e710e159785513a6572e6d01f944d39",
            "mask.zip": "5d2b0800e92cda7f9a083f2081d75381",
            "s1_part1.zip": "8796a047261bc1f986ce2f9843b5a292",
            "s1_part2.zip": "d69f357ad5bc2b5fdae31581183e3789",
            "s2_temporal_subset_part1.zip": "9b2c24a65bcd5505126395858577ea3d",
            "s2_temporal_subset_part2.zip": "10ed0134ad62ecf6d978687eea2f31fb",
            "s2_autumn_part1.zip": "0e97b20efea4ee4edb9ba06484a895f0",
            "s2_autumn_part2.zip": "e999924025a08253f3847d634fe55d45",
            "s2_spring_part1.zip": "3c3613dcd372f1900b5894236283c89b",
            "s2_spring_part2.zip": "458e50ea3fb6c7865406eb635220ba55",
            "s2_summer_part1.zip": "628448d26388d2daebc5eefd3284431d",
            "s2_summer_part2.zip": "fd38bd42bddad741901f43c2cfbe7261",
            "s2_winter_part1.zip": "e3877d58dd63196f48d400eaede8bb09",
            "s2_winter_part2.zip": "158143afcb7771fe3bb5259fa60137bb",
        }
        monkeypatch.setattr(MapInWild, "md5s", md5s)
        root = os.path.join("tests", "data", "mapinwild")
        transforms = nn.Identity()
        modality = ["mask", "esa_wc", "viirs"]
        return MapInWild(
            root, modality=modality, transforms=transforms, download=True, checksum=True
        )  # noqa: E501

    def test_getitem(self, dataset: MapInWild) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].ndim == 3

    def test_len(self, dataset: MapInWild) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: MapInWild) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            MapInWild(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            MapInWild(root=str(tmp_path), download=False)

    def test_plot(self, dataset: MapInWild) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="prediction")
        plt.close()
