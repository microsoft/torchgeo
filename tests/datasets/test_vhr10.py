# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import VHR10, DatasetNotFoundError

pytest.importorskip("pycocotools")


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestVHR10:
    @pytest.fixture(params=["positive", "negative"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> VHR10:
        pytest.importorskip("rarfile", minversion="4")
        monkeypatch.setattr(torchgeo.datasets.vhr10, "download_url", download_url)
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        url = os.path.join("tests", "data", "vhr10", "NWPU VHR-10 dataset.rar")
        monkeypatch.setitem(VHR10.image_meta, "url", url)
        md5 = "5fddb0dfd56a80638831df9f90cbf37a"
        monkeypatch.setitem(VHR10.image_meta, "md5", md5)
        url = os.path.join("tests", "data", "vhr10", "annotations.json")
        monkeypatch.setitem(VHR10.target_meta, "url", url)
        md5 = "833899cce369168e0d4ee420dac326dc"
        monkeypatch.setitem(VHR10.target_meta, "md5", md5)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return VHR10(root, split, transforms, download=True, checksum=True)

    @pytest.fixture
    def mock_missing_modules(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in {"pycocotools.coco", "skimage.measure"}:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_getitem(self, dataset: VHR10) -> None:
        for i in range(2):
            x = dataset[i]
            assert isinstance(x, dict)
            assert isinstance(x["image"], torch.Tensor)
            if dataset.split == "positive":
                assert isinstance(x["labels"], torch.Tensor)
                assert isinstance(x["boxes"], torch.Tensor)
                if "masks" in x:
                    assert isinstance(x["masks"], torch.Tensor)

    def test_len(self, dataset: VHR10) -> None:
        if dataset.split == "positive":
            assert len(dataset) == 5
        elif dataset.split == "negative":
            assert len(dataset) == 150

    def test_add(self, dataset: VHR10) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        if dataset.split == "positive":
            assert len(ds) == 10
        elif dataset.split == "negative":
            assert len(ds) == 300

    def test_already_downloaded(self, dataset: VHR10) -> None:
        VHR10(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            VHR10(split="train")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            VHR10(str(tmp_path))

    def test_mock_missing_module(
        self, dataset: VHR10, mock_missing_modules: None
    ) -> None:
        if dataset.split == "positive":
            with pytest.raises(
                ImportError,
                match="pycocotools is not installed and is required to use this datase",
            ):
                VHR10(dataset.root, dataset.split)

            with pytest.raises(
                ImportError,
                match="scikit-image is not installed and is required to plot masks",
            ):
                x = dataset[0]
                dataset.plot(x)

    def test_plot(self, dataset: VHR10) -> None:
        pytest.importorskip("skimage", minversion="0.18")
        x = dataset[1].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        if dataset.split == "positive":
            scores = [0.7, 0.3, 0.7]
            for i in range(3):
                x = dataset[i]
                x["prediction_labels"] = x["labels"]
                x["prediction_boxes"] = x["boxes"]
                x["prediction_scores"] = torch.Tensor([scores[i]])
                if "masks" in x:
                    x["prediction_masks"] = x["masks"]
                    dataset.plot(x, show_feats="masks")
                    plt.close()
