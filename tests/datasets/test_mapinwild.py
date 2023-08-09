# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import MapInWild

pytest.importorskip("pandas", minversion="1.1.3")


class TestMapInWild:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(self, monkeypatch: MonkeyPatch) -> MapInWild:
        md5s = {
            "ESA_WC.zip": "3acbc2a2bc299e5a967f9349db484872",
            "VIIRS.zip": "25081c93534c42c28d70ae67469c33ac",
            "mask.zip": "e5042709de9ee97f83d14b3bb0a7bb78",
            "s1_part1.zip": "c4732b3b9239983634dec2066fda11cf",
            "s1_part2.zip": "50b72b5470ec6801e969cf292fec7d1d",
            "s2_autumn_part1.zip": "2e4b7b09202504d1dc95c83a65685d8a",
            "s2_autumn_part2.zip": "b8882dd9290124e1de4f0e0872774e6a",
            "s2_spring_part1.zip": "c29184039ad7e7aee4a4bcae4e013bca",
            "s2_spring_part2.zip": "9cb5a81804344c080591592828ba0a22",
            "s2_summer_part1.zip": "8ce599ff71a9bfa2445cf460284fbec8",
            "s2_summer_part2.zip": "7f5643b88c4b7395bb97f428d9573190",
            "s2_temporal_subset_part1.zip": "89fcd50a65a7cfbbac61d91af3b44cb7",
            "s2_temporal_subset_part2.zip": "5aa028759cefcb8b4fbe3da95e0f1ff1",
            "s2_winter_part1.zip": "83a72480a8be070f8b256458e9a1a4f8",
            "s2_winter_part2.zip": "fddd797d8bfd932e31a36a8423ff3704",
        }

        monkeypatch.setattr(MapInWild, "md5s", md5s)
        root = os.path.join("tests", "data", "mapinwild")
        transforms = nn.Identity()
        modality = [
            "mask",
            "viirs",
            "esa_wc",
            "s2_winter",
            "s1",
            "s2_summer",
            "s2_spring",
            "s2_autumn",
            "s2_temporal_subset",
        ]  # noqa: E501
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
            MapInWild(root=str(tmp_path))

    def test_download(self) -> None:
        MapInWild.modality_urls["test_coverage"] = {
            "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/test_coverage.zip"  # noqa: E501
        }
        MapInWild.md5s["test_coverage.zip"] = "612bc89e728c71d3347e5406cf6cfb3f"
        root = os.path.join("tests", "data", "mapinwild")
        modality = ["test_coverage"]
        MapInWild(root, modality=modality, download=True, checksum=True)  # noqa: E501

    def test_corrupted(self, tmp_path: Path) -> None:
        root = os.path.join("tests", "data", "mapinwild")
        with open(os.path.join(root, "test_coverage.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            MapInWild(root=root, download=False, checksum=True)

    def test_plot(self, dataset: MapInWild) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="prediction")
        plt.close()

        images = [
            torch.rand(1, 64, 64),
            torch.rand(2, 64, 64),
            torch.rand(3, 64, 64),
            torch.rand(4, 64, 64),
        ]
        esa_wc_values = torch.arange(0, 110, 10)
        esa_wc_image = torch.tile(esa_wc_values, (1, 22, 2))
        mask = torch.rand(1, 64, 64)

        sample_1 = {"image": images[0], "mask": mask}
        sample_2 = {"image": images[1], "mask": mask}
        sample_3 = {"image": images[2], "mask": mask}
        sample_4 = {"image": images[3], "mask": mask}
        sample_5 = {"image": esa_wc_image, "mask": mask}

        dataset.plot(sample_1)
        plt.close()
        dataset.plot(sample_2)
        plt.close()
        dataset.plot(sample_3)
        plt.close()
        dataset.plot(sample_4)
        plt.close()
        dataset.plot(sample_5)
        plt.close()
