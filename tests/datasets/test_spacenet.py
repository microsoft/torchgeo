# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import SpaceNet1, SpaceNet2, SpaceNet4, SpaceNet7

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
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return SpaceNet1(
            root, image=request.param, transforms=transforms, download=True, api_key=""
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
        transforms = nn.Identity()  # type: ignore[attr-defined]
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


class TestSpaceNet4:
    @pytest.fixture(params=["PAN", "MS", "PS-RGBNIR"])
    def dataset(
        self,
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> SpaceNet4:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(  # type: ignore[attr-defined]
            radiant_mlhub.Collection, "fetch", fetch_collection
        )
        test_md5 = {"sn4_AOI_6_Atlanta": "ea37c2d87e2c3a1d8b2a7c2230080d46"}

        test_angles = ["nadir", "off-nadir", "very-off-nadir"]

        monkeypatch.setattr(  # type: ignore[attr-defined]
            SpaceNet4, "collection_md5_dict", test_md5
        )
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return SpaceNet4(
            root,
            image=request.param,
            angles=test_angles,
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet4) -> None:
        # Get image-label pair with empty label to
        # ensure coverage
        x = dataset[2]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "PS-RGBNIR":
            assert x["image"].shape[0] == 4
        elif dataset.image == "MS":
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet4) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: SpaceNet4) -> None:
        SpaceNet4(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet4(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet4) -> None:
        dataset.collection_md5_dict["sn4_AOI_6_Atlanta"] = "randommd5hash123"
        with pytest.raises(
            RuntimeError, match="Collection sn4_AOI_6_Atlanta corrupted"
        ):
            SpaceNet4(root=dataset.root, download=True, checksum=True)


class TestSpaceNet7:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self,
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> SpaceNet7:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(  # type: ignore[attr-defined]
            radiant_mlhub.Collection, "fetch", fetch_collection
        )
        test_md5 = {
            "sn7_train_source": "254fd6b16e350b071137b2658332091f",
            "sn7_train_labels": "05befe86b037a3af75c7143553033664",
            "sn7_test_source": "37d98d44a9da39657ed4b7beee22a21e",
        }

        monkeypatch.setattr(  # type: ignore[attr-defined]
            SpaceNet7, "collection_md5_dict", test_md5
        )
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return SpaceNet7(
            root, split=request.param, transforms=transforms, download=True, api_key=""
        )

    def test_getitem(self, dataset: SpaceNet7) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        if dataset.split == "train":
            assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: SpaceNet7) -> None:
        if dataset.split == "train":
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: SpaceNet4) -> None:
        SpaceNet7(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet7(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet4) -> None:
        dataset.collection_md5_dict["sn7_train_source"] = "randommd5hash123"
        with pytest.raises(RuntimeError, match="Collection sn7_train_source corrupted"):
            SpaceNet7(root=dataset.root, download=True, checksum=True)
