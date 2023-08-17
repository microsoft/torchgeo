# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import MapInWild

pytest.importorskip("pandas", minversion="1.1.3")


class TestMapInWild:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(self, monkeypatch: MonkeyPatch) -> MapInWild:
        md5s = {
            "ESA_WC.zip": "59f19a93127430779a1f52a995888fc1",
            "VIIRS.zip": "56d197ebe2a254521b32e1ba7621b88a",
            "mask.zip": "eb2a7fa28b216176064aab92a8e6c22e",
            "s1_part1.zip": "737b595fd6a3d457d25f294fc6eb19b5",
            "s1_part2.zip": "889a9dd0664b08e7351e507ea96e633e",
            "s2_autumn_part1.zip": "5dddd1a4c16a08051b2dbd95096e1092",
            "s2_autumn_part2.zip": "bb41dd5fb097d9a5f99e088367bb0625",
            "s2_spring_part1.zip": "60736b95f243015f265b3757e21408b7",
            "s2_spring_part2.zip": "9b953323ea15061b35152bb82a1a6e74",
            "s2_summer_part1.zip": "e69ca5746d8f5aa682f1871ab0f82596",
            "s2_summer_part2.zip": "4f98247b175ccbafb55ef07975f90614",
            "s2_temporal_subset_part1.zip": "cf51a14951c14a2699386b223ba03ec0",
            "s2_temporal_subset_part2.zip": "1a776938fe6f302fddb608a44385802b",
            "s2_winter_part1.zip": "e2e9509825fb71fe7f38a4941cd73a42",
            "s2_winter_part2.zip": "f6db6cdd941c316dd5f23b4898558b9e",
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
        ]
        return MapInWild(
            root, modality=modality, transforms=transforms, download=True, checksum=True
        )

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
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automatically download the dataset."
        with pytest.raises(RuntimeError, match=err):
            MapInWild(str(tmp_path), download=False, checksum=True)

    def test_download(self) -> None:
        url = "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/"
        MapInWild.modality_urls["test_coverage"] = {
            os.path.join(url, "test_coverage.zip")
        }
        MapInWild.modality_urls["split_file"] = {
            os.path.join(url, "split_IDs/split_IDs.csv")
        }

        MapInWild.md5s["test_coverage.zip"] = "612bc89e728c71d3347e5406cf6cfb3f"
        root = os.path.join("tests", "data", "mapinwild")
        modality = ["test_coverage", "split_file"]
        MapInWild(root, modality=modality, download=True, checksum=True)

    def test_corrupted(self) -> None:
        root = os.path.join("tests", "data", "mapinwild")
        with open(os.path.join(root, "test_coverage.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            MapInWild(root=root, download=True, checksum=True)

    @pytest.fixture(params=["pandas"])
    def mock_missing_module(self, monkeypatch: MonkeyPatch, request: SubRequest) -> str:
        import_orig = builtins.__import__
        package = str(request.param)

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        return package

    def test_mock_missing_module(
        self, dataset: MapInWild, mock_missing_module: str
    ) -> None:
        package = mock_missing_module
        if package == "pandas":
            with pytest.raises(
                ImportError,
                match=f"{package} is not installed and is required to use this dataset",
            ):
                MapInWild(dataset.root)

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
