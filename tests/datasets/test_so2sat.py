# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, RGBBandsMissingError, So2Sat

pytest.importorskip('h5py', minversion='3')


class TestSo2Sat:
    @pytest.fixture(params=['train', 'validation', 'test'])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> So2Sat:
        md5s_by_version = {
            '2': {
                'train': '56e6fa0edb25b065124a3113372f76e5',
                'validation': '940c95a737bd2fcdcc46c9a52b31424d',
                'test': 'e97a6746aadc731a1854097f32ab1755',
            }
        }

        monkeypatch.setattr(So2Sat, 'md5s_by_version', md5s_by_version)
        root = os.path.join('tests', 'data', 'so2sat')
        split = request.param
        transforms = nn.Identity()
        return So2Sat(root=root, split=split, transforms=transforms, checksum=True)

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == 'h5py':
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mocked_import)

    def test_getitem(self, dataset: So2Sat) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_len(self, dataset: So2Sat) -> None:
        assert len(dataset) == 2

    def test_out_of_bounds(self, dataset: So2Sat) -> None:
        # h5py at version 2.10.0 raises a ValueError instead of an IndexError so we
        # check for both here
        with pytest.raises((IndexError, ValueError)):
            dataset[2]

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            So2Sat(split='foo')

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            So2Sat(bands=('OK', 'BK'))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            So2Sat(str(tmp_path))

    def test_plot(self, dataset: So2Sat) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_rgb(self, dataset: So2Sat) -> None:
        dataset = So2Sat(root=dataset.root, bands=('S2_B03',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0], suptitle='Single Band')

    def test_mock_missing_module(
        self, dataset: So2Sat, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match='h5py is not installed and is required to use this dataset',
        ):
            So2Sat(dataset.root)
