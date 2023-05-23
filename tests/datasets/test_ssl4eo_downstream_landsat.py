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

import torchgeo.datasets.utils
from torchgeo.datasets import SSL4EODownstream


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSSL4EODownstream:
    @pytest.fixture(params=product([("l5-l1", 2011)], ["cdl", "nlcd"]))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SSL4EODownstream:
        root = str(tmp_path)
        sensor_year, mask_product = request.param
        input_sensor, year = sensor_year

        img_dir = os.path.join(
            "tests",
            "data",
            "ssl4eo_downstream_landsat",
            input_sensor,
            f"ssl4eo-{input_sensor}-conus",
        )
        mask_dir = os.path.join(
            "tests",
            "data",
            "ssl4eo_downstream_landsat",
            input_sensor,
            f"{input_sensor.split('-')[0]}-{mask_product}-{year}",
        )

        shutil.copy(img_dir, root)
        shutil.copy(mask_dir, root)

        transforms = nn.Identity()
        return SSL4EODownstream(
            root=root,
            input_sensor=input_sensor,
            mask_product=mask_product,
            split=split,
            download=True,
            checksum=True,
            transforms=transforms,
        )

    def test_getitem(self, dataset: SSL4EODownstream) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EODownstream(split="foo")

    def test_invalid_input_sensor(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EODownstream(split="foo")

    def test_invalid_mask_product(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EODownstream(split="foo")

    def test_add(self, dataset: SSL4EODownstream) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
