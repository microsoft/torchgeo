# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import glob
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import MapInWild

pytest.importorskip("pandas", minversion="1.1.3")


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestMapInWild:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> MapInWild:
        monkeypatch.setattr(torchgeo.datasets.mapinwild, "download_url", download_url)

        md5s = {
            "ESA_WC.zip": "b14b6b54ef33ffd6e5802fb3096b0710",
            "VIIRS.zip": "012734f798a39fdddf24fd9bdfc5f790",
            "mask.zip": "d7fcb0ac6fc51a0189c60a8c44403194",
            "s1_part1.zip": "17a9e23df5063d066098c84a4074cef1",
            "s1_part2.zip": "c6748e547f18288300d670e70bff234a",
            "s2_autumn_part1.zip": "89890200ef79bd80c6cbf1d4bbd314c6",
            "s2_autumn_part2.zip": "22b7f6be3e59daaf264a179c94498e8e",
            "s2_spring_part1.zip": "5f91cd21d3929b1e384b128b90f4efe3",
            "s2_spring_part2.zip": "5ab77f7352d42cc5de76bf7293e20877",
            "s2_summer_part1.zip": "d80b22a94c5fa0aca61fdc9d8cc7c361",
            "s2_summer_part2.zip": "2ec2f215fcc8247df64f50288f664a8b",
            "s2_temporal_subset_part1.zip": "0398bc292d9c3e8513444c96c5b0da96",
            "s2_temporal_subset_part2.zip": "c063386289476dc2342872abb7c57f1c",
            "s2_winter_part1.zip": "4f67835837f865f03cda9841b0da0717",
            "s2_winter_part2.zip": "b6654fb29a315bddaca8b3e7a9f34930",
        }
        monkeypatch.setattr(MapInWild, "md5s", md5s)

        urls = os.path.join("tests", "data", "mapinwild")

        modality_urls = {
            "esa_wc": {os.path.join(urls, "ESA_WC.zip")},
            "viirs": {os.path.join(urls, "VIIRS.zip")},
            "mask": {os.path.join(urls, "mask.zip")},
            "s1": {
                os.path.join(urls, "s1_part1.zip"),
                os.path.join(urls, "s1_part2.zip"),
            },
            "s2_temporal_subset": {
                os.path.join(urls, "s2_temporal_subset_part1.zip"),
                os.path.join(urls, "s2_temporal_subset_part2.zip"),
            },
            "s2_autumn": {
                os.path.join(urls, "s2_autumn_part1.zip"),
                os.path.join(urls, "s2_autumn_part2.zip"),
            },
            "s2_spring": {
                os.path.join(urls, "s2_spring_part1.zip"),
                os.path.join(urls, "s2_spring_part2.zip"),
            },
            "s2_summer": {
                os.path.join(urls, "s2_summer_part1.zip"),
                os.path.join(urls, "s2_summer_part2.zip"),
            },
            "s2_winter": {
                os.path.join(urls, "s2_winter_part1.zip"),
                os.path.join(urls, "s2_winter_part2.zip"),
            },
            "split_ids": {os.path.join(urls, "split_IDs.csv")},
        }

        monkeypatch.setattr(MapInWild, "modality_urls", modality_urls)

        root = str(tmp_path)

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
        with pytest.raises(RuntimeError, match="Dataset not found"):
            MapInWild(root=str(tmp_path))

    def test_not_extracted(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "mapinwild", "*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        splitfile = os.path.join("tests", "data", "mapinwild", "split_IDs.csv")
        shutil.copy(splitfile, root)
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
        MapInWild(root, modality=modality)

    def test_corrupted(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "mapinwild", "*.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        splitfile = os.path.join("tests", "data", "mapinwild", "split_IDs.csv")
        shutil.copy(splitfile, root)
        with open(os.path.join(tmp_path, "mask.zip"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            MapInWild(root=str(tmp_path), checksum=True)

    def test_already_downloaded(self, dataset: MapInWild, tmp_path: Path) -> None:
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
        MapInWild(root=str(tmp_path), modality=modality, download=True)

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
