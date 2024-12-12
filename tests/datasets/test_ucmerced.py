# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import DatasetNotFoundError, UCMerced


class TestUCMerced:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> UCMerced:
        url = os.path.join('tests', 'data', 'ucmerced') + os.sep
        monkeypatch.setattr(UCMerced, 'url', url)
        split = request.param
        transforms = nn.Identity()
        return UCMerced(tmp_path, split, transforms, download=True)

    def test_getitem(self, dataset: UCMerced) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_len(self, dataset: UCMerced) -> None:
        assert len(dataset) == 4

    def test_add(self, dataset: UCMerced) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 8

    def test_already_downloaded(self, dataset: UCMerced, tmp_path: Path) -> None:
        UCMerced(tmp_path)

    def test_already_downloaded_not_extracted(
        self, dataset: UCMerced, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        shutil.copy(dataset.url + dataset.filename, tmp_path)
        UCMerced(tmp_path)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            UCMerced(tmp_path)

    def test_plot(self, dataset: UCMerced) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
