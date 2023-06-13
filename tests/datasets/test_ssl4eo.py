# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import SSL4EOL, SSL4EOS12


class TestSSL4EOL:
    @pytest.fixture(params=zip(SSL4EOL.metadata.keys(), [1, 1, 2, 2, 4]))
    def dataset(self, request: SubRequest) -> SSL4EOL:
        split, seasons = request.param
        root = os.path.join("tests", "data", "ssl4eo", "l", split)
        transforms = nn.Identity()
        return SSL4EOL(root, split, seasons, transforms)

    def test_getitem(self, dataset: SSL4EOL) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert (
            x["image"].size(0)
            == dataset.seasons * dataset.metadata[dataset.split]["num_bands"]
        )

    def test_len(self, dataset: SSL4EOL) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: SSL4EOL) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2 * 2

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOL(split="foo")

    def test_plot(self, dataset: SSL4EOL) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle="Test")
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()


class TestSSL4EOS12:
    @pytest.fixture(params=zip(SSL4EOS12.metadata.keys(), [1, 2, 4]))
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> SSL4EOS12:
        monkeypatch.setitem(
            SSL4EOS12.metadata["s1"], "md5", "a716f353e4c2f0014f2e1f1ad848f82e"
        )
        monkeypatch.setitem(
            SSL4EOS12.metadata["s2c"], "md5", "85eaf474af5642588a97dc5c991cfc15"
        )
        monkeypatch.setitem(
            SSL4EOS12.metadata["s2a"], "md5", "df41a5d1ae6f840bc9a11ee254110369"
        )

        root = os.path.join("tests", "data", "ssl4eo", "s12")
        split, seasons = request.param
        transforms = nn.Identity()
        return SSL4EOS12(root, split, seasons, transforms, checksum=True)

    def test_getitem(self, dataset: SSL4EOS12) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].size(0) == dataset.seasons * len(dataset.bands)

    def test_len(self, dataset: SSL4EOS12) -> None:
        assert len(dataset) == 251079

    def test_add(self, dataset: SSL4EOS12) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2 * 251079

    def test_extract(self, tmp_path: Path) -> None:
        for split in SSL4EOS12.metadata:
            filename = SSL4EOS12.metadata[split]["filename"]
            shutil.copyfile(
                os.path.join("tests", "data", "ssl4eo", "s12", filename),
                tmp_path / filename,
            )
        SSL4EOS12(str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOS12(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SSL4EOS12(str(tmp_path))

    def test_plot(self, dataset: SSL4EOS12) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle="Test")
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
