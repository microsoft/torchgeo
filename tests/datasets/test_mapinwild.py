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
            "ESA_WC.zip": "65e1276f3357a3b1c8d4966e3cdc5f49",
            "VIIRS.zip": "f85af612bde060b48f683d199291d690",
            "mask.zip": "7388351116da4efa7c47fe99f390f040",
            "s1_part1.zip": "dcd039f2d6befcd42bbb369c37dfc19b",
            "s1_part2.zip": "e442796f1283516414c769bb9d1f02c7",
            "s2_autumn_part1.zip": "b4ddb4d23112f59b3d37d6f98d05e8e0",
            "s2_autumn_part2.zip": "b5bfb0b03757944b363e54e626a72c3b",
            "s2_spring_part1.zip": "63e1ebc52e9bd039080f2e573e9ef080",
            "s2_spring_part2.zip": "d52b1375de8eb128a83f38b8ae3e3046",
            "s2_summer_part1.zip": "360f6c5bccbda67b5bd1cb0b2f3ef2fa",
            "s2_summer_part2.zip": "8e746d0fb15ef0214752da2081052c42",
            "s2_temporal_subset_part1.zip": "9de26bd9e8825af00675482759f89f21",
            "s2_temporal_subset_part2.zip": "0fa26dceb5fac1bc1dd4484319bebbbd",
            "s2_winter_part1.zip": "b19eeef2c7dec1cb0e2cf5e2fa684918",
            "s2_winter_part2.zip": "2151695dd6f096ebb55e4ea0bb08c882",
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
