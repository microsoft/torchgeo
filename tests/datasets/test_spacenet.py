# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import (
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet7,
)

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
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet1:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {"sn1_AOI_1_RIO": "829652022c2df4511ee4ae05bc290250"}

        # Refer https://github.com/python/mypy/issues/1032
        monkeypatch.setattr(SpaceNet1, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()
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

    def test_plot(self, dataset: SpaceNet1) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()


class TestSpaceNet2:
    @pytest.fixture(params=["PAN", "MS", "PS-MS", "PS-RGB"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet2:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn2_AOI_2_Vegas": "6ceae7ff8c557346e8a4c8b6c61cc1b9",
            "sn2_AOI_3_Paris": "811e6a26fdeb8be445fed99769fa52c5",
            "sn2_AOI_4_Shanghai": "139d1627d184c74426a85ad0222f7355",
            "sn2_AOI_5_Khartoum": "435535120414b74165aa87f051c3a2b3",
        }

        monkeypatch.setattr(SpaceNet2, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()
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

    def test_plot(self, dataset: SpaceNet2) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()


class TestSpaceNet3:
    @pytest.fixture(params=zip(["PAN", "MS"], [False, True]))
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet3:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn3_AOI_3_Paris": "197440e0ade970169a801a173a492c27",
            "sn3_AOI_5_Khartoum": "b21ff7dd33a15ec32bd380c083263cdf",
        }

        monkeypatch.setattr(SpaceNet3, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()
        return SpaceNet3(
            root,
            image=request.param[0],
            speed_mask=request.param[1],
            collections=["sn3_AOI_3_Paris", "sn3_AOI_5_Khartoum"],
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet3) -> None:
        # Iterate over all elements to maximize coverage
        samples = [dataset[i] for i in range(len(dataset))]
        x = samples[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "MS":
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet3) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: SpaceNet3) -> None:
        SpaceNet3(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet3(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet3) -> None:
        dataset.collection_md5_dict["sn3_AOI_5_Khartoum"] = "randommd5hash123"
        with pytest.raises(
            RuntimeError, match="Collection sn3_AOI_5_Khartoum corrupted"
        ):
            SpaceNet3(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet3) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        dataset.plot({"image": x["image"]})
        plt.close()


class TestSpaceNet4:
    @pytest.fixture(params=["PAN", "MS", "PS-RGBNIR"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet4:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {"sn4_AOI_6_Atlanta": "ea37c2d87e2c3a1d8b2a7c2230080d46"}

        test_angles = ["nadir", "off-nadir", "very-off-nadir"]

        monkeypatch.setattr(SpaceNet4, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()
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

    def test_plot(self, dataset: SpaceNet4) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()


class TestSpaceNet5:
    @pytest.fixture(params=zip(["PAN", "MS"], [False, True]))
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet5:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn5_AOI_7_Moscow": "e0d5f41f1b6b0ee7696c15e5ff3141f5",
            "sn5_AOI_8_Mumbai": "ab898700ee586a137af492b84a08e662",
        }

        monkeypatch.setattr(SpaceNet5, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()
        return SpaceNet5(
            root,
            image=request.param[0],
            speed_mask=request.param[1],
            collections=["sn5_AOI_7_Moscow", "sn5_AOI_8_Mumbai"],
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet5) -> None:
        # Iterate over all elements to maximize coverage
        samples = [dataset[i] for i in range(len(dataset))]
        x = samples[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "MS":
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet5) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: SpaceNet5) -> None:
        SpaceNet5(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet5(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet5) -> None:
        dataset.collection_md5_dict["sn5_AOI_8_Mumbai"] = "randommd5hash123"
        with pytest.raises(RuntimeError, match="Collection sn5_AOI_8_Mumbai corrupted"):
            SpaceNet5(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet5) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        dataset.plot({"image": x["image"]})
        plt.close()


class TestSpaceNet7:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet7:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn7_train_source": "254fd6b16e350b071137b2658332091f",
            "sn7_train_labels": "05befe86b037a3af75c7143553033664",
            "sn7_test_source": "37d98d44a9da39657ed4b7beee22a21e",
        }

        monkeypatch.setattr(SpaceNet7, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()
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

    def test_plot(self, dataset: SpaceNet7) -> None:
        x = dataset[0].copy()
        if dataset.split == "train":
            x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
