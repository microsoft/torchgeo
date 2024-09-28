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

from torchgeo.datasets import DatasetNotFoundError, FieldsOfTheWorld

pytest.importorskip('pyarrow', minversion='15.0.0')


class TestFieldsOfTheWorld:
    @pytest.fixture(
        params=product(['train', 'val', 'test'], ['2-class', '3-class', 'instance'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FieldsOfTheWorld:
        split, task = request.param

        monkeypatch.setattr(FieldsOfTheWorld, 'valid_countries', ['austria'])
        monkeypatch.setattr(
            FieldsOfTheWorld,
            'country_to_md5',
            {'austria': '1cf9593c9bdceeaba21bbcb24d35816c'},
        )
        base_url = os.path.join('tests', 'data', 'ftw') + '/'
        monkeypatch.setattr(FieldsOfTheWorld, 'base_url', base_url)
        root = tmp_path
        transforms = nn.Identity()
        return FieldsOfTheWorld(
            root,
            split,
            task,
            countries='austria',
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: FieldsOfTheWorld) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: FieldsOfTheWorld) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: FieldsOfTheWorld) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: FieldsOfTheWorld) -> None:
        FieldsOfTheWorld(root=dataset.root, download=True)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'ftw', 'austria.zip')
        root = tmp_path
        shutil.copy(url, root)
        FieldsOfTheWorld(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FieldsOfTheWorld(tmp_path)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            FieldsOfTheWorld(split='foo')

    def test_plot(self, dataset: FieldsOfTheWorld) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
