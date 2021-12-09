# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import NASAMarineDebris, NASAMarineDebrisDataModule


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join("tests", "data", "nasa_marine_debris", "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestNASAMarineDebris:
    @pytest.fixture()
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
    ) -> NASAMarineDebris:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(  # type: ignore[attr-defined]
            radiant_mlhub.Dataset, "fetch", fetch
        )
        md5s = ["fe8698d1e68b3f24f0b86b04419a797d", "d8084f5a72778349e07ac90ec1e1d990"]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            NASAMarineDebris, "md5s", md5s
        )
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return NASAMarineDebris(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["boxes"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["boxes"].shape[-1] == 4

    def test_len(self, dataset: NASAMarineDebris) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(
        self, dataset: NASAMarineDebris, tmp_path: Path
    ) -> None:
        NASAMarineDebris(root=str(tmp_path), download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: NASAMarineDebris, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        os.makedirs(str(tmp_path), exist_ok=True)
        Dataset().download(output_dir=str(tmp_path))
        print(os.listdir(str(tmp_path)))
        NASAMarineDebris(root=str(tmp_path), download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = rf"Dataset not found in `root={str(tmp_path)}` directory and "
        "`download=False`, either specify a different `root` directory"
        "or use `download=True` to automaticaly download the dataset."
        with pytest.raises(RuntimeError, match=err):
            NASAMarineDebris(str(tmp_path))

    def test_plot(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction_boxes"] = x["boxes"].clone()
        dataset.plot(x)
        plt.close()


class TestNASAMarineDebrisDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> NASAMarineDebrisDataModule:
        root = os.path.join("tests", "data", "nasa_marine_debris")
        batch_size = 2
        num_workers = 0
        val_split_pct = 0.3
        test_split_pct = 0.3
        dm = NASAMarineDebrisDataModule(
            root, batch_size, num_workers, val_split_pct, test_split_pct
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: NASAMarineDebrisDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: NASAMarineDebrisDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: NASAMarineDebrisDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
