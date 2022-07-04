# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import SEN12MS


class TestSEN12MS:
    @pytest.fixture(params=["train", "test"])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> SEN12MS:
        md5s = [
            "b7d9e183a460979e997b443517a78ded",
            "7131dbb098c832fff84c2b8a0c8f1126",
            "b1057fea6ced6d648e5b16efeac352ad",
            "2da32111fcfb80939aea7b18c2250fa8",
            "c688ad6475660dbdbc36f66a1dd07da7",
            "2ecd0dce2a21372513955c604b07e24f",
            "dbc84c03edf77a68f789a6f7d2ea66a9",
            "3e42a7dc4bb1ecd8c588930bf49b5c2b",
            "c29053cb8cf5d75e333b1b51d37f62fe",
            "5b6880526bc6da488154092741392042",
            "d1b51c39b1013f2779fecf1f362f6c28",
            "078def1e13ce4e88632d65f5c73a6259",
            "02d5128ac1fc2bf8762091b4f319762d",
            "02d5128ac1fc2bf8762091b4f319762d",
        ]

        monkeypatch.setattr(SEN12MS, "md5s", md5s)
        root = os.path.join("tests", "data", "sen12ms")
        split = request.param
        transforms = nn.Identity()
        return SEN12MS(root, split, transforms=transforms, checksum=True)

    def test_getitem(self, dataset: SEN12MS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 15

    def test_len(self, dataset: SEN12MS) -> None:
        assert len(dataset) == 8

    def test_add(self, dataset: SEN12MS) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 16

    def test_out_of_bounds(self, dataset: SEN12MS) -> None:
        with pytest.raises(IndexError):
            dataset[8]

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SEN12MS(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            SEN12MS(str(tmp_path), checksum=True)

        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            SEN12MS(str(tmp_path), checksum=False)

    def test_check_integrity_light(self) -> None:
        root = os.path.join("tests", "data", "sen12ms")
        ds = SEN12MS(root, checksum=False)
        assert isinstance(ds, SEN12MS)

    def test_band_subsets(self) -> None:
        root = os.path.join("tests", "data", "sen12ms")
        for bands in SEN12MS.BAND_SETS.values():
            ds = SEN12MS(root, bands=bands, checksum=False)
            x = ds[0]["image"]
            assert x.shape[0] == len(bands)

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            SEN12MS(bands=("OK", "BK"))

    def test_plot(self, dataset: SEN12MS) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="prediction")
        plt.close()

    def test_plot_rgb(self, dataset: SEN12MS) -> None:
        dataset = SEN12MS(root=dataset.root, bands=("B03",))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle="Single Band")
