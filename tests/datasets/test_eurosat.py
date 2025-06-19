# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    DatasetNotFoundError,
    EuroSAT,
    EuroSAT100,
    EuroSATSpatial,
    RGBBandsMissingError,
)


class TestEuroSAT:
    @pytest.fixture(
        params=product([EuroSAT, EuroSATSpatial, EuroSAT100], ['train', 'val', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> EuroSAT:
        base_class: type[EuroSAT] = request.param[0]
        split: str = request.param[1]
        url = os.path.join('tests', 'data', 'eurosat') + os.sep
        monkeypatch.setattr(base_class, 'url', url)
        transforms = nn.Identity()
        return base_class(tmp_path, split=split, transforms=transforms, download=True)

    def test_getitem(self, dataset: EuroSAT) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            EuroSAT(split='foo')

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            EuroSAT(bands=('OK', 'BK'))

    def test_len(self, dataset: EuroSAT) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: EuroSAT) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: EuroSAT, tmp_path: Path) -> None:
        type(dataset)(tmp_path, split=dataset.split)

    def test_already_downloaded_not_extracted(
        self, dataset: EuroSAT, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        shutil.copy(dataset.url + dataset.filename, tmp_path)
        type(dataset)(tmp_path, split=dataset.split)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EuroSAT(tmp_path)

    def test_missing_images_and_zip_no_download(self, tmp_path: Path) -> None:
        """Ensure DatasetNotFoundError is raised if images and zip are missing and download=False."""
        split_file = tmp_path / 'eurosat-train.txt'
        split_file.write_text('dummy.tif\n')
        with pytest.raises(DatasetNotFoundError):
            EuroSAT(root=tmp_path, split='train', download=False)

    def test_plot(self, dataset: EuroSAT) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_rgb(self, dataset: EuroSAT, tmp_path: Path) -> None:
        dataset = type(dataset)(tmp_path, split=dataset.split, bands=('B03',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0], suptitle='Single Band')
