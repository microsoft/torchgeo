# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import SSL4EOLBenchmark


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSSL4EOLBenchmark:
    @pytest.fixture(
        params=product(
            ["tm_toa", "etm_toa", "etm_sr", "oli_tirs_toa", "oli_sr"],
            ["cdl", "nlcd"],
            ["train", "val", "test"],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SSL4EOLBenchmark:
        monkeypatch.setattr(
            torchgeo.datasets.ssl4eo_benchmark, "download_url", download_url
        )
        root = str(tmp_path)

        url = os.path.join("tests", "data", "ssl4eo_benchmark_landsat", "{}.tar.gz")
        monkeypatch.setattr(SSL4EOLBenchmark, "url", url)

        input_sensor, mask_product, split = request.param
        monkeypatch.setattr(
            SSL4EOLBenchmark, "split_percentages", [1 / 3, 1 / 3, 1 / 3]
        )

        img_md5s = {
            "tm_toa": "27f0562206baec86c5fdd1d7f069ef91",
            "etm_toa": "0350f83c8462a64ffd192d8ebe070842",
            "etm_sr": "277e1657b89e141fa3085fd01053162d",
            "oli_tirs_toa": "53350e7ee0616df47859d28a29e170da",
            "oli_sr": "8235bcce500657b9e0cfcb3af6bb1480",
        }
        monkeypatch.setattr(SSL4EOLBenchmark, "img_md5s", img_md5s)

        mask_md5s = {
            "tm": {
                "cdl": "762104b3fc41afe1ef63f5ea80940d4b",
                "nlcd": "57391b79a33ccd482471b377ae2de7f1",
            },
            "etm": {
                "cdl": "8285e0d051081a9379cd150c7669971e",
                "nlcd": "916f4a433df6c8abca15b45b60d005d3",
            },
            "oli": {
                "cdl": "729a7b75b8749c8a7f26e5ece164e73f",
                "nlcd": "e237adcee8b43d4eca86a6d169ae2761",
            },
        }
        monkeypatch.setattr(SSL4EOLBenchmark, "mask_md5s", mask_md5s)

        transforms = nn.Identity()
        return SSL4EOLBenchmark(
            root=root,
            input_sensor=input_sensor,
            mask_product=mask_product,
            split=split,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: SSL4EOLBenchmark) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(split="foo")

    def test_invalid_input_sensor(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(input_sensor="foo")

    def test_invalid_mask_product(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(mask_product="foo")

    def test_add(self, dataset: SSL4EOLBenchmark) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: SSL4EOLBenchmark) -> None:
        SSL4EOLBenchmark(
            root=dataset.root,
            input_sensor=dataset.input_sensor,
            mask_product=dataset.mask_product,
            download=True,
        )

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "ssl4eo_benchmark_landsat", "*.tar.gz")
        root = str(tmp_path)
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        SSL4EOLBenchmark(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SSL4EOLBenchmark(str(tmp_path))

    def test_plot(self, dataset: SSL4EOLBenchmark) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle="Test")
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample)
        plt.close()
