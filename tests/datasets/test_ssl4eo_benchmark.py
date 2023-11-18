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
from torchgeo.datasets import (
    CDL,
    NLCD,
    DatasetNotFoundError,
    RasterDataset,
    SSL4EOLBenchmark,
)


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

        sensor, product, split = request.param
        monkeypatch.setattr(
            SSL4EOLBenchmark, "split_percentages", [1 / 3, 1 / 3, 1 / 3]
        )

        img_md5s = {
            "tm_toa": "ecfdd3dcbc812c5e7cf272a5cddb33e9",
            "etm_sr": "3e598245948eb7d072d5b83c95f22422",
            "etm_toa": "e24ff11f6aedb3930380b53cb6f780b6",
            "oli_tirs_toa": "490baa1eedd5032277e2a07f45dd8c2b",
            "oli_sr": "884f6e28a23a1b7d464eff39abd7667d",
        }
        monkeypatch.setattr(SSL4EOLBenchmark, "img_md5s", img_md5s)

        mask_md5s = {
            "tm": {
                "cdl": "43f30648e0f7c8dba78fa729b6db9ffe",
                "nlcd": "4272958acb32cc3b83f593684bc3e63c",
            },
            "etm": {
                "cdl": "b215b7e3b65b18a6d52ce9a35c90a16f",
                "nlcd": "f823fc69965d7f6215f52bea2141df41",
            },
            "oli": {
                "cdl": "aaa956d7aa985e8de2c565858c9ac4e8",
                "nlcd": "cc49207df010a4f358fb16a46772e9ae",
            },
        }
        monkeypatch.setattr(SSL4EOLBenchmark, "mask_md5s", mask_md5s)

        transforms = nn.Identity()
        return SSL4EOLBenchmark(
            root=root,
            sensor=sensor,
            product=product,
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

    @pytest.mark.parametrize("product,base_class", [("nlcd", NLCD), ("cdl", CDL)])
    def test_classes(self, product: str, base_class: RasterDataset) -> None:
        root = os.path.join("tests", "data", "ssl4eo_benchmark_landsat")
        classes = list(base_class.cmap.keys())[:5]
        ds = SSL4EOLBenchmark(root, product=product, classes=classes)
        sample = ds[0]
        mask = sample["mask"]
        assert mask.max() < len(classes)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(split="foo")

    def test_invalid_sensor(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(sensor="foo")

    def test_invalid_product(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(product="foo")

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(classes=[-1])

        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(classes=[11])

    def test_add(self, dataset: SSL4EOLBenchmark) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: SSL4EOLBenchmark) -> None:
        SSL4EOLBenchmark(
            root=dataset.root,
            sensor=dataset.sensor,
            product=dataset.product,
            download=True,
        )

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "ssl4eo_benchmark_landsat", "*.tar.gz")
        root = str(tmp_path)
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        SSL4EOLBenchmark(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
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
