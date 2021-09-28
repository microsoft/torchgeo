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

from torchgeo.datasets import SpaceNet1, SpaceNet2
from torchgeo.transforms import Identity

TEST_DATA_DIR = "tests/data/spacenet"


class Collection:
    def __init__(self, collection_id: str) -> None:
        self.collection_id = collection_id

    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(TEST_DATA_DIR, "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch_collection(collection_id: str, **kwargs: str) -> Collection:
    return Collection(collection_id)


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
            radiant_mlhub.Collection, "fetch", fetch_collection
        )
        test_md5 = {"sn1_AOI_1_RIO": "829652022c2df4511ee4ae05bc290250"}

        # Refer https://github.com/python/mypy/issues/1032
        monkeypatch.setattr(  # type: ignore[attr-defined]
            SpaceNet1, "collection_md5_dict", test_md5
        )
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


class TestSpaceNet2:
    @pytest.fixture(params=["PAN", "MS", "PS-MS", "PS-RGB"])
    def dataset(
        self,
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> SpaceNet2:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(  # type: ignore[attr-defined]
            radiant_mlhub.Collection, "fetch", fetch_collection
        )
        test_md5 = {
            "sn2_AOI_2_Vegas": "b3236f58604a9d746c4e09b3e487e427",
            "sn2_AOI_3_Paris": "811e6a26fdeb8be445fed99769fa52c5",
            "sn2_AOI_4_Shanghai": "139d1627d184c74426a85ad0222f7355",
            "sn2_AOI_5_Khartoum": "435535120414b74165aa87f051c3a2b3",
        }

        monkeypatch.setattr(  # type: ignore[attr-defined]
            SpaceNet2, "collection_md5_dict", test_md5
        )
        root = str(tmp_path)
        transforms = Identity()
        return SpaceNet2(
            root,
            image=request.param,
            collections=["sn2_AOI_2_Vegas", "sn2_AOI_5_Khartoum"],
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "PS-RGB":
            assert x["image"].shape[0] == 3
        elif dataset.image in ["MS", "PS-MS"]:
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    # TODO: Change len to 4 when radiantearth/radiant-mlhub#65 is fixed
    def test_len(self, dataset: SpaceNet2) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: SpaceNet2) -> None:
        SpaceNet2(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet2(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet2) -> None:
        dataset.collection_md5_dict["sn2_AOI_2_Vegas"] = "randommd5hash123"
        with pytest.raises(RuntimeError, match="Collection sn2_AOI_2_Vegas corrupted"):
            SpaceNet2(root=dataset.root, download=True, checksum=True)
