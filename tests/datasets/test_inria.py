# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, InriaAerialImageLabeling


class TestInriaAerialImageLabeling:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch
    ) -> InriaAerialImageLabeling:
        root = os.path.join("tests", "data", "inria")
        test_md5 = "3ecbe95eb84aea064e455c4321546be1"
        monkeypatch.setattr(InriaAerialImageLabeling, "md5", test_md5)
        transforms = nn.Identity()
        return InriaAerialImageLabeling(
            root, split=request.param, transforms=transforms, checksum=True
        )

    def test_getitem(self, dataset: InriaAerialImageLabeling) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        if dataset.split == "train":
            assert isinstance(x["mask"], torch.Tensor)
            assert x["mask"].ndim == 2
        assert x["image"].shape[0] == 3
        assert x["image"].ndim == 3

    def test_len(self, dataset: InriaAerialImageLabeling) -> None:
        if dataset.split == "train":
            assert len(dataset) == 2
        elif dataset.split == "val":
            assert len(dataset) == 5
        elif dataset.split == "test":
            assert len(dataset) == 7

    def test_already_downloaded(self, dataset: InriaAerialImageLabeling) -> None:
        InriaAerialImageLabeling(root=dataset.root)

    def test_not_downloaded(self, tmp_path: str) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            InriaAerialImageLabeling(str(tmp_path))

    def test_dataset_checksum(self, dataset: InriaAerialImageLabeling) -> None:
        InriaAerialImageLabeling.md5 = "randommd5hash123"
        shutil.rmtree(os.path.join(dataset.root, dataset.directory))
        with pytest.raises(RuntimeError, match="Dataset corrupted"):
            InriaAerialImageLabeling(root=dataset.root, checksum=True)

    def test_plot(self, dataset: InriaAerialImageLabeling) -> None:
        x = dataset[0].copy()
        if dataset.split == "train":
            x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
