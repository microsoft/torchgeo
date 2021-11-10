# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import FAIR1M, FAIR1MDataModule


class TestFAIR1M:
    @pytest.fixture
    def dataset(
        self,
        tmp_path: Path,
    ) -> FAIR1M:
        data_dir = os.path.join("tests", "data", "fair1m")
        shutil.copytree(
            os.path.join(data_dir, "images"), os.path.join(str(tmp_path), "images")
        )
        shutil.copytree(
            os.path.join(data_dir, "labelXmls"),
            os.path.join(str(tmp_path), "labelXmls"),
        )
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return FAIR1M(root, transforms)

    def test_getitem(self, dataset: FAIR1M) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["bbox"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["bbox"].shape[-2:] == (5, 2)
        assert x["label"].ndim == 1

    def test_len(self, dataset: FAIR1M) -> None:
        assert len(dataset) == 4


class TestFAIR1MDataModule:
    @pytest.fixture(scope="class", params=[True, False])
    def datamodule(self, request: SubRequest) -> FAIR1MDataModule:
        root = os.path.join("tests", "data", "fair1m")
        batch_size = 2
        num_workers = 0
        unsupervised_mode = request.param
        dm = FAIR1MDataModule(
            root,
            batch_size,
            num_workers,
            val_split_pct=0.33,
            test_split_pct=0.33,
            unsupervised_mode=unsupervised_mode,
        )
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
