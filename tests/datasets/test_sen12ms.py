# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Generator

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
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], request: SubRequest
    ) -> SEN12MS:
        md5s = [
            "85650b52b6c5de282547f837275c0739",
            "c6b181319295bc088883591dd549c065",
            "7906ec529abad3f7d6f7dbb4517c6e2e",
            "257b33b835223a9d033aae0dcd309e1a",
            "26117a8c6f27838f98c8961ab35294e8",
            "87933d453a15431437697f4701b4d8ed",
            "9be07f30e582d11d5529afbd0c623c5a",
            "73323197968407645ec155d91fcb4ce9",
            "035c9f2edae2e9a0e37c8d63ed4baef3",
            "164f940c205965c2b8e41794ccdeeaf5",
            "b104631c369e98b96c11a5af5456e9ea",
            "c00fc28dc0f36a2ee4a448e01701b58b",
            "02d5128ac1fc2bf8762091b4f319762d",
            "02d5128ac1fc2bf8762091b4f319762d",
        ]

        monkeypatch.setattr(SEN12MS, "md5s", md5s)  # type: ignore[attr-defined]
        root = os.path.join("tests", "data", "sen12ms")
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
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
