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
            "ESA_WC.zip": "1e958600753dbc4d9726f24b59d0939a",
            "VIIRS.zip": "1d9dd53836f1e352567739051ec0af75",
            "mask.zip": "2ce0b1b799adf8a087578c650474c83c",
            "s1_part1.zip": "40f7cd6a30917a24fd0b8f1daa94221b",
            "s1_part2.zip": "fc85146053ddf74a6b7404ade73a048d",
            "s2_temporal_subset_part1.zip": "43a001401a655fabee0817f4079233a0",
            "s2_temporal_subset_part2.zip": "16ee6a63fc486b84a95f6e314bb4adeb",
            "s2_autumn_part1.zip": "1dd661731df64b3277a3d6628a7d4d8b",
            "s2_autumn_part2.zip": "cf1bddb2061156a3b40fcef706c8b9f2",
            "s2_spring_part1.zip": "79e43ed71dc9b402d3f2952858cae82a",
            "s2_spring_part2.zip": "bab7d10f9ce68bf6637d26e9ba98b6b4",
            "s2_summer_part1.zip": "b6dbc51ea990dbcacc504f4af0d40496",
            "s2_summer_part2.zip": "66031600753d905bdcbdb88e3c3635df",
            "s2_winter_part1.zip": "2a1edc59ae2570b50f255550c1f006da",
            "s2_winter_part2.zip": "c374e48f00cc4da45a54a02e56dc925d",
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

    def test_download(self) -> None:
        MapInWild.modality_urls["test_coverage"] = {
            "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/test_coverage.zip"  # noqa: E501
        }
        MapInWild.md5s["test_coverage.zip"] = "612bc89e728c71d3347e5406cf6cfb3f"
        root = os.path.join("tests", "data", "mapinwild")
        modality = ["test_coverage"]
        MapInWild(root, modality=modality, download=True, checksum=True)  # noqa: E501
