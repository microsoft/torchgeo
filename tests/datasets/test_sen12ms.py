# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import SEN12MS
from torchgeo.transforms import Identity


class TestSEN12MS:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], request: SubRequest
    ) -> SEN12MS:
        md5s = [
            "7f14be13d3f62c09b4dd5b4d55c97fd6",
            "48182d44b375360381f36d432956b225",
            "96cf1b8405d4149c6fe61ad7100bd65d",
            "ba8e7e10fba9eea6900ddc530c86025a",
            "7ba7c51f2fb3a2074b7bbd3e24f9d70d",
            "280c9be2d1e13e663824dccd85e1e42f",
            "a5284baf48534d4bc77acb1b103ff16c",
            "c6b176fed0cdd5033cb1835506e40ee4",
            "adc672746b79be4c4edc8b1a564e3ff4",
            "194fab4a4e067a0452824c4e39f61b77",
            "7899c0c36c884ae8c991ab8518b0d177",
            "ccfee543d4351bcc5aa68729e8cc795c",
        ]

        monkeypatch.setattr(SEN12MS, "md5s", md5s)  # type: ignore[attr-defined]
        root = os.path.join("tests", "data", "sen12ms")
        split = request.param
        transforms = Identity()
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
