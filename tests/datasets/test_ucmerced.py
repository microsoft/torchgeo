# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.utils
from torchgeo.datasets import UCMerced


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestUCMerced:
    @pytest.fixture()
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> UCMerced:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.ucmerced, "download_url", download_url
        )
        md5 = "95e710774f3ef6d9ecb0cd42e4d0fc23"
        monkeypatch.setattr(UCMerced, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "ucmerced", "UCMerced_LandUse.zip")
        monkeypatch.setattr(UCMerced, "url", url)  # type: ignore[attr-defined]
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return UCMerced(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: UCMerced) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)

    def test_len(self, dataset: UCMerced) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: UCMerced) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: UCMerced, tmp_path: Path) -> None:
        UCMerced(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: UCMerced, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        download_url(dataset.url, root=str(tmp_path))
        UCMerced(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory and `download=False`, "
        "either specify a different `root` directory or use `download=True` "
        "to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            UCMerced(str(tmp_path))
