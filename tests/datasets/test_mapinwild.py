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
            "ESA_WC.zip": "561ec0294bc19a9de105d1b4d81dd7d7",
            "VIIRS.zip": "96c631706677f1e2ccd53d9443148afd",
            "mask.zip": "26aa75f768b25e632d507b083a575485",
            "s1_part1.zip": "7ea9ed1fa6caa088ab1457f0aba31fc6",
            "s1_part2.zip": "c6c115cd0f9546772f49b33a07205d43",
            "s2_autumn_part1.zip": "5558f28502d52545cd77403dd0f81328",
            "s2_autumn_part2.zip": "352fbb9e2dd298a3ab231f4177f387e0",
            "s2_spring_part1.zip": "0ea18b7fdd701264f202f8575c23eea2",
            "s2_spring_part2.zip": "4c2a90bb63d56aa5076410de636972a7",
            "s2_summer_part1.zip": "9935c4d1c2c7c4bea9a83eed05bf292d",
            "s2_summer_part2.zip": "f8b7a499d2897c803534b345ff1324ef",
            "s2_temporal_subset_part1.zip": "595c84774a4408e4605053e01b9db8e9",
            "s2_temporal_subset_part2.zip": "6795d76e4e1dac437fba4c552af8f9b1",
            "s2_winter_part1.zip": "14f766820472e89af7c9de4fb91d2fa4",
            "s2_winter_part2.zip": "8a7094192368b0bc0c355649f10857c3",
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
        err = "Dataset not found in `root={self.root}` directory and `download=False`, "
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
