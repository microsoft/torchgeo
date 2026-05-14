# Copyright (c) TorchGeo Contributors. All rights reserved.
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

from torchgeo.datasets import PASTIS, PASTIS100, DatasetNotFoundError


class TestPASTIS:
    @pytest.fixture(
        params=product(
            [PASTIS, PASTIS100],
            [
                {'folds': (1, 2), 'bands': PASTIS.s2_bands, 'mode': 'semantic'},
                {'folds': (1, 2), 'bands': ('B04', 'B03', 'B02'), 'mode': 'semantic'},
                {'folds': (1, 2), 'bands': PASTIS.s1a_bands, 'mode': 'semantic'},
                {'folds': (1, 2), 'bands': PASTIS.s1d_bands, 'mode': 'instance'},
            ],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> PASTIS:
        base_class: type[PASTIS] = request.param[0]
        params: dict[str, str | tuple[str, ...]] = request.param[1]

        root = tmp_path
        bands = params['bands']
        mode = params['mode']
        assert isinstance(mode, str)
        transforms = nn.Identity()

        url = os.path.join('tests', 'data', 'pastis', 'PASTIS-R.zip')
        monkeypatch.setattr(base_class, 'url', url)
        return base_class(
            root=root,
            folds=(1, 2),
            bands=bands,
            mode=mode,
            transforms=transforms,
            download=True,
        )

    def test_getitem_semantic(self, dataset: PASTIS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_getitem_instance(self, dataset: PASTIS) -> None:
        dataset.mode = 'instance'
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_len(self, dataset: PASTIS) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: PASTIS) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_extracted(self, dataset: PASTIS) -> None:
        type(dataset)(
            root=dataset.root,
            folds=dataset.folds,
            bands=dataset.bands,
            mode=dataset.mode,
            download=True,
        )

    def test_already_downloaded(self, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'pastis', 'PASTIS-R.zip')
        root = tmp_path
        shutil.copy(url, root)
        PASTIS(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            PASTIS(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'PASTIS-R.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            PASTIS(root=tmp_path, checksum=True)

    def test_invalid_fold(self) -> None:
        with pytest.raises(AssertionError):
            PASTIS(folds=(0,))

    def test_invalid_mode(self) -> None:
        with pytest.raises(AssertionError):
            PASTIS(mode='invalid')

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError, match='bands must be a subset of'):
            PASTIS(bands=('B01',))

    def test_invalid_bands_empty(self) -> None:
        with pytest.raises(ValueError, match='bands must not be empty'):
            PASTIS(bands=())

    def test_plot(self, dataset: PASTIS) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        if dataset.mode == 'instance':
            x['prediction_labels'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
