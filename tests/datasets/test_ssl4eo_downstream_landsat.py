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
    @pytest.fixture(
        params=product(
            ["l7-l1", "l7-l2", "l8-l1", "l8-l2"],
            ["cdl", "nlcd"],
            ["train", "val", "test"],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SSL4EODownstream:
        input_sensor, mask_product, split = request.param
        monkeypatch.setattr(torchgeo.datasets.eurosat, "download_url", download_url)

        root = str(tmp_path)
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
