# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import Vaihingen, VaihingenDataModule


class TestVaihingen:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], request: SubRequest
    ) -> Vaihingen:
        md5s = ["c15fbff78d307e51c73f609c0859afc3", "0cb795003a01154a72db7efaabbc76ae"]
        splits = {
            "train": ["top_mosaic_09cm_area1.tif", "top_mosaic_09cm_area11.tif"],
            "test": ["top_mosaic_09cm_area6.tif", "top_mosaic_09cm_area24.tif"],
        }
        monkeypatch.setattr(Vaihingen, "md5s", md5s)  # type: ignore[attr-defined]
        monkeypatch.setattr(Vaihingen, "splits", splits)  # type: ignore[attr-defined]
        root = os.path.join("tests", "data", "vaihingen")
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return Vaihingen(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: Vaihingen) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: Vaihingen) -> None:
        assert len(dataset) == 2

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join("tests", "data", "vaihingen")
        filenames = [
            "ISPRS_semantic_labeling_Vaihingen.zip",
            "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip"
        ]
        for filename in filenames:
            shutil.copyfile(
                os.path.join(root, filename), os.path.join(str(tmp_path), filename)
            )
        Vaihingen(root=str(tmp_path))

    def test_corrupted(self, tmp_path: Path) -> None:
        filenames = [
            "ISPRS_semantic_labeling_Vaihingen.zip",
            "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip"
        ]
        for filename in filenames:
            with open(os.path.join(tmp_path, filename), "w") as f:
                f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            Vaihingen(root=str(tmp_path), checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            Vaihingen(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in `root` directory"):
            Vaihingen(str(tmp_path))

    def test_plot(self, dataset: Vaihingen) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        dataset.plot(x, show_titles=False)
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)


class TestVaihingenDataModule:
    @pytest.fixture(scope="class", params=[0.0, 0.5])
    def datamodule(self, request: SubRequest) -> VaihingenDataModule:
        root = os.path.join("tests", "data", "vaihingen")
        batch_size = 1
        num_workers = 0
        val_split_size = request.param
        dm = VaihingenDataModule(
            root, batch_size, num_workers, val_split_pct=val_split_size
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: VaihingenDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: VaihingenDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: VaihingenDataModule) -> None:
        next(iter(datamodule.test_dataloader()))