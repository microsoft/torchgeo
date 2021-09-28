# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import EuroSAT
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestEuroSAT:
    @pytest.fixture()
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> EuroSAT:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "aa051207b0547daba0ac6af57808d68e"
        monkeypatch.setattr(EuroSAT, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "eurosat", "EuroSATallBands.zip")
        monkeypatch.setattr(EuroSAT, "url", url)  # type: ignore[attr-defined]

        monkeypatch.setattr(  # type: ignore[attr-defined]
            EuroSAT,
            "class_counts",
            {
                "AnnualCrop": 1,
                "Forest": 1,
                "HerbaceousVegetation": 0,
                "Highway": 0,
                "Industrial": 0,
                "Pasture": 0,
                "PermanentCrop": 0,
                "Residential": 0,
                "River": 0,
                "SeaLake": 0,
            },
        )

        root = str(tmp_path)
        transforms = Identity()
        return EuroSAT(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: EuroSAT) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: EuroSAT) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: EuroSAT) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: EuroSAT) -> None:
        EuroSAT(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            EuroSAT(str(tmp_path))
