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

import torchgeo.datasets.utils
from torchgeo.datasets import CropHarvest


def download_url(url: str, root: str, filename: str, md5: str) -> None:
    shutil.copy(url, os.path.join(root, filename))


def download_and_extract_archive(url: str, root: str, filename: str, md5: str) -> None:
    download_url(url, root, filename, md5)
    shutil.unpack_archive(
        os.path.join(root, filename), os.path.join(root, "CropHarvest")
    )


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
        assert isinstance(x["data"], torch.Tensor)
        assert isinstance(x["label"], str)
        assert x["data"].shape == (12, 18)
        y = dataset[2]
        assert y["label"] == "Some"

    def test_len(self, dataset: CropHarvest) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: CropHarvest, tmp_path: Path) -> None:
        CropHarvest(root=str(tmp_path), download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(
            RuntimeError,
            match="Dataset not found or corrupted. "
            + "You can use download=True to download it",
        ):
            CropHarvest(str(tmp_path))

    def test_plot(self, dataset: CropHarvest) -> None:
        x = dataset[0].copy()
        dataset.plot(x, subtitle="Test")
        plt.close()
