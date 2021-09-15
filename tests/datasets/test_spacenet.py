# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import SpaceNet1
from torchgeo.transforms import Identity

TEST_DATA_DIR = "tests/data/spacenet"


class Dataset:
    def __init__(self, collection_id: str) -> None:
        self.collection_id = collection_id

    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(TEST_DATA_DIR, self.collection_id, "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(collection_id: str, **kwargs: str) -> Dataset:
    return Dataset(collection_id)


class TestSpaceNet1:
    @pytest.fixture(params=["rgb", "8band"])
    def dataset(
        self,
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> SpaceNet1:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(  # type: ignore[attr-defined]
            radiant_mlhub.Dataset, "fetch", fetch
        )
        test_md5 = "829652022c2df4511ee4ae05bc290250"
        monkeypatch.setattr(SpaceNet1, "md5", test_md5)  # type: ignore[attr-defined]
        root = str(tmp_path)
        transforms = Identity()
        return SpaceNet1(
            root,
            image=request.param,
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet1) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "rgb":
            assert x["image"].shape[0] == 3
        else:
            assert x["image"].shape[0] == 8

    def test_len(self, dataset: SpaceNet1) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: SpaceNet1) -> None:
        SpaceNet1(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet1(str(tmp_path))
