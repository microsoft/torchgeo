# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import CopernicusBench, DatasetNotFoundError


class TestCopernicusBench:
    @pytest.fixture(params=[('cloud_s2', 'l1_cloud_s2')])
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

    def test_missing(self, dataset: CopernicusBench, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CopernicusBench(dataset.name, tmp_path)
