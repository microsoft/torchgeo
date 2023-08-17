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
            "ESA_WC.zip": "7654ef46c6523ad61d2c56fa97773512",
            "VIIRS.zip": "d8ad5736203d9253a7ee08b552e33556",
            "mask.zip": "ce5a7da8ec0753476b9e21bc04ded9bd",
            "s1_part1.zip": "a9735c8a947bc9cdef61cf6a45f99a26",
            "s1_part2.zip": "3f705e97dfea46a013b113add94f2c63",
            "s2_autumn_part1.zip": "8dafc79b11daa2d17d7d7335348b39e7",
            "s2_autumn_part2.zip": "d8be0cce59d146246f3e7c7826db725a",
            "s2_spring_part1.zip": "551159d8781bf261d5a1b7c53920f35e",
            "s2_spring_part2.zip": "71def6e55e321f17b4db5e2832e0842a",
            "s2_summer_part1.zip": "71f5c9c0213a2227a1f7fb9bca7e3f35",
            "s2_summer_part2.zip": "f205b7b99df1f1380901524c58b4e5f8",
            "s2_temporal_subset_part1.zip": "be2b9cd5fbdcf399c9d34899314cdccb",
            "s2_temporal_subset_part2.zip": "44ab31e20e4fbd89a35f73cab623a739",
            "s2_winter_part1.zip": "f4b4524e6d9a45b5ed3801ba160cf00c",
            "s2_winter_part2.zip": "b07dcefd0ca520c74424c71fb9f36774",
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
