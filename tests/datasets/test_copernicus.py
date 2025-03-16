# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch

from torchgeo.datasets import (
    CopernicusBench,
    DatasetNotFoundError,
    RGBBandsMissingError,
)


class TestCopernicusBench:
    @pytest.fixture(params=[('cloud_s2', 'l1_cloud_s2'), ('cloud_s3', 'l1_cloud_s3')])
    def dataset(self, request: SubRequest) -> CopernicusBench:
        dataset, directory = request.param
        root = os.path.join('tests', 'data', 'copernicus', directory)
        transforms = nn.Identity()
        return CopernicusBench(dataset, root, transforms=transforms)

    def test_getitem(self, dataset: CopernicusBench) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: CopernicusBench) -> None:
        assert len(dataset) == 1

    def test_extract(self, dataset: CopernicusBench, tmp_path: Path) -> None:
        root = dataset.root
        file = dataset.zipfile
        shutil.copyfile(os.path.join(root, file), tmp_path / file)
        CopernicusBench(dataset.name, tmp_path)

    def test_download(
        self, dataset: CopernicusBench, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        url = os.path.join(dataset.root, dataset.zipfile)
        monkeypatch.setattr(dataset.dataset.__class__, 'url', url)
        CopernicusBench(dataset.name, tmp_path, download=True)

    def test_not_downloaded(self, dataset: CopernicusBench, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CopernicusBench(dataset.name, tmp_path)

    def test_plot(self, dataset: CopernicusBench) -> None:
        x = dataset[0]
        if 'mask' in x:
            x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_not_rgb(self, dataset: CopernicusBench) -> None:
        all_bands = list(dataset.all_bands)
        rgb_bands = list(dataset.rgb_bands)
        for band in rgb_bands:
            all_bands.remove(band)

        dataset = CopernicusBench(dataset.name, dataset.root, bands=all_bands)
        match = 'Dataset does not contain some of the RGB bands'
        with pytest.raises(RGBBandsMissingError, match=match):
            dataset.plot(dataset[0])
