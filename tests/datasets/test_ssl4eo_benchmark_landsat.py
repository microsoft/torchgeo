# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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

from torchgeo.datasets import SSL4EOLBenchmark


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSSL4EOLBenchmark:
    @pytest.fixture(
        params=product(
            ["tm_toa", "etm_toa", "etm_sr", "oli_tirs_toa", "oli_sr"], ["cdl", "nlcd"]
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SSL4EOLBenchmark:
        root = str(tmp_path)
        input_sensor, mask_product = request.param

        img_dir = os.path.join("tests", "data", "ssl4eo_benchmark_landsat")
        shutil.copytree(img_dir, root, dirs_exist_ok=True)

        transforms = nn.Identity()
        return SSL4EOLBenchmark(
            root=root,
            input_sensor=input_sensor,
            mask_product=mask_product,
            transforms=transforms,
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
            SSL4EOLBenchmark(split="foo")

    def test_invalid_mask_product(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(split="foo")

    def test_add(self, dataset: SSL4EOLBenchmark) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_plot(self, dataset: SSL4EOLBenchmark) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle="Test")
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample)
        plt.close()
