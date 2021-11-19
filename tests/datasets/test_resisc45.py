# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import sys
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import RESISC45, RESISC45DataModule

pytest.importorskip("rarfile")


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
class TestRESISC45:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> RESISC45:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.resisc45, "download_url", download_url
        )
        md5 = "5895dea3757ba88707d52f5521c444d3"
        monkeypatch.setattr(RESISC45, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "resisc45", "NWPU-RESISC45.rar")
        monkeypatch.setattr(RESISC45, "url", url)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            RESISC45,
            "split_urls",
            {
                "train": os.path.join(
                    "tests", "data", "resisc45", "resisc45-train.txt"
                ),
                "val": os.path.join("tests", "data", "resisc45", "resisc45-val.txt"),
                "test": os.path.join("tests", "data", "resisc45", "resisc45-test.txt"),
            },
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            RESISC45,
            "split_md5s",
            {
                "train": "7760b1960c9a3ff46fb985810815e14d",
                "val": "7760b1960c9a3ff46fb985810815e14d",
                "test": "7760b1960c9a3ff46fb985810815e14d",
            },
        )
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return RESISC45(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: RESISC45) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3

    def test_len(self, dataset: RESISC45) -> None:
        assert len(dataset) == 9

    def test_already_downloaded(self, dataset: RESISC45, tmp_path: Path) -> None:
        RESISC45(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: RESISC45, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        RESISC45(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            RESISC45(str(tmp_path))

    def test_plot(self, dataset: RESISC45) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["label"].clone()
        dataset.plot(x)
        plt.close()


class TestRESISC45DataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> RESISC45DataModule:
        root = os.path.join("tests", "data", "resisc45")
        batch_size = 2
        num_workers = 0
        dm = RESISC45DataModule(root, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: RESISC45DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: RESISC45DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: RESISC45DataModule) -> None:
        next(iter(datamodule.test_dataloader()))
