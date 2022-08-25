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
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import BeninSmallHolderCashews


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join("tests", "data", "ts_cashew_benin", "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestBeninSmallHolderCashews:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> BeninSmallHolderCashews:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)
        source_md5 = "255efff0f03bc6322470949a09bc76db"
        labels_md5 = "ed2195d93ca6822d48eb02bc3e81c127"
        monkeypatch.setitem(BeninSmallHolderCashews.image_meta, "md5", source_md5)
        monkeypatch.setitem(BeninSmallHolderCashews.target_meta, "md5", labels_md5)
        monkeypatch.setattr(BeninSmallHolderCashews, "dates", ("2019_11_05",))
        root = str(tmp_path)
        transforms = nn.Identity()
        bands = BeninSmallHolderCashews.ALL_BANDS

        return BeninSmallHolderCashews(
            root,
            transforms=transforms,
            bands=bands,
            download=True,
            api_key="",
            checksum=True,
            verbose=True,
        )

    def test_getitem(self, dataset: BeninSmallHolderCashews) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert isinstance(x["x"], torch.Tensor)
        assert isinstance(x["y"], torch.Tensor)

    def test_len(self, dataset: BeninSmallHolderCashews) -> None:
        assert len(dataset) == 72

    def test_add(self, dataset: BeninSmallHolderCashews) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 144

    def test_already_downloaded(self, dataset: BeninSmallHolderCashews) -> None:
        BeninSmallHolderCashews(root=dataset.root, download=True, api_key="")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            BeninSmallHolderCashews(str(tmp_path))

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            BeninSmallHolderCashews(bands=["B01", "B02"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="is an invalid band name."):
            BeninSmallHolderCashews(bands=("foo", "bar"))

    def test_plot(self, dataset: BeninSmallHolderCashews) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()

    def test_failed_plot(self, dataset: BeninSmallHolderCashews) -> None:
        single_band_dataset = BeninSmallHolderCashews(root=dataset.root, bands=("B01",))
        with pytest.raises(ValueError, match="Dataset doesn't contain"):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle="Test")
