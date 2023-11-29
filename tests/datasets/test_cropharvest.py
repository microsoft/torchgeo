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

import torchgeo.datasets.utils
from torchgeo.datasets import CropHarvest, DatasetNotFoundError

pytest.importorskip("h5py", minversion="3")


def download_url(url: str, root: str, filename: str, md5: str) -> None:
    shutil.copy(url, os.path.join(root, filename))


def download_and_extract_archive(url: str, root: str, filename: str, md5: str) -> None:
    download_url(url, root, filename, md5)
    shutil.unpack_archive(os.path.join(root, filename), os.path.join(root))


class TestCropHarvest:
    file_dict = {
        "features": {
            "url": os.path.join(
                "tests", "data", "cropharvest", "CropHarvest", "features.tar.gz"
            ),
            "filename": "features.tar.gz",
            "extracted_filename": os.path.join("features", "arrays"),
            "md5": "cad4df655c75caac805a80435e46ee3e",
        },
        "labels": {
            "url": os.path.join(
                "tests", "data", "cropharvest", "CropHarvest", "labels.geojson"
            ),
            "filename": "labels.geojson",
            "extracted_filename": "labels.geojson",
            "md5": "bf7bae6812fc7213481aff6a2e34517d",
        },
    }

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "h5py":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> CropHarvest:
        monkeypatch.setattr(
            torchgeo.datasets.cropharvest,
            "download_and_extract_archive",
            download_and_extract_archive,
        )
        monkeypatch.setattr(torchgeo.datasets.cropharvest, "download_url", download_url)
        monkeypatch.setattr(CropHarvest, "file_dict", self.file_dict)

        root = str(tmp_path)
        transforms = nn.Identity()
        os.makedirs(os.path.join(root, "CropHarvest"))
        dataset = CropHarvest(root, transforms, download=True, checksum=True)
        print(dataset.files)
        return dataset

    def test_getitem(self, dataset: CropHarvest) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["array"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["array"].shape == (12, 18)
        y = dataset[2]
        assert y["label"] == 1

    def test_len(self, dataset: CropHarvest) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: CropHarvest, tmp_path: Path) -> None:
        CropHarvest(root=str(tmp_path), download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            CropHarvest(str(tmp_path))

    def test_plot(self, dataset: CropHarvest) -> None:
        x = dataset[0].copy()
        dataset.plot(x, subtitle="Test")
        plt.close()

    def test_mock_missing_module(
        self, dataset: CropHarvest, tmp_path: Path, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="h5py is not installed and is required to use this dataset",
        ):
            CropHarvest(root=str(tmp_path), download=True)[0]
