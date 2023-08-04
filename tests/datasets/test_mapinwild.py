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
            "ESA_WC.zip": "67b84b3eee87b4a46327c2e84791a23e",
            "VIIRS.zip": "27f049d84b965337556494057070146d",
            "mask.zip": "8d4a78cd6f4082363c91f33f768e148f",
            "s1_part1.zip": "20ecc05d7bff42beaadb8a4c73904e8e",
            "s1_part2.zip": "32953389ef20aec07b15b8c60bbc4734",
            "s2_temporal_subset_part1.zip": "b3df47e7b594eb0c57422960ed20733e",
            "s2_temporal_subset_part2.zip": "873ba42f86ea836a236b83464e770a64",
            "s2_autumn_part1.zip": "36870866b4180be7cbf24371a0c94a06",
            "s2_autumn_part2.zip": "e2a6c6387be6fdd78f03b4a773584dec",
            "s2_spring_part1.zip": "83fa9af070818a0be43490928be93d06",
            "s2_spring_part2.zip": "ad194b8085c9086ea1ecd9a2032a88ec",
            "s2_summer_part1.zip": "cc4ed45c005942f1a43ccce9a8171073",
            "s2_summer_part2.zip": "76e4ea90a4e36cf197549cf112cb5680",
            "s2_winter_part1.zip": "5f7db8e5e9db2f61727803461c8dda76",
            "s2_winter_part2.zip": "14be1c9721a8aaafcc4126e54c2b2c34",
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
