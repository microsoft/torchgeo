# Copyright (c) TorchGeo Contributors. All rights reserved.
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

from torchgeo.datasets import DLRSD, DatasetNotFoundError, DLRSDBase, DLRSDMultilabel


class TestDLRSD:
    @pytest.fixture(params=[DLRSD, DLRSDMultilabel])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DLRSDBase:
        base_class: type[DLRSDBase] = request.param
        url = os.path.join('tests', 'data', 'dlrsd') + os.sep
        monkeypatch.setattr(base_class, 'url', url)
        transforms = nn.Identity()
        return base_class(tmp_path, transforms, download=True)

    def test_getitem(self, dataset: DLRSDBase) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        if isinstance(dataset, DLRSDMultilabel):
            assert isinstance(x['label'], torch.Tensor)
            assert x['label'].shape == (17,)
        else:
            assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: DLRSDBase) -> None:
        assert len(dataset) == 4

    def test_add(self, dataset: DLRSDBase) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 8

    def test_already_downloaded(self, dataset: DLRSDBase, tmp_path: Path) -> None:
        type(dataset)(tmp_path)

    def test_already_downloaded_not_extracted(
        self, dataset: DLRSDBase, tmp_path: Path
    ) -> None:
        shutil.rmtree(os.path.join(tmp_path, dataset.directory))
        type(dataset)(tmp_path)

    @pytest.mark.parametrize('base_class', [DLRSD, DLRSDMultilabel])
    def test_corrupted(self, tmp_path: Path, base_class: type[DLRSDBase]) -> None:
        with open(os.path.join(tmp_path, 'DLRSD.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            base_class(root=tmp_path, checksum=True)

    @pytest.mark.parametrize('base_class', [DLRSD, DLRSDMultilabel])
    def test_not_downloaded(self, tmp_path: Path, base_class: type[DLRSDBase]) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            base_class(tmp_path)

    def test_plot(self, dataset: DLRSDBase) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        target_key = 'label' if isinstance(dataset, DLRSDMultilabel) else 'mask'
        x['prediction'] = x[target_key].clone()
        dataset.plot(x)
        plt.close()
        if isinstance(dataset, DLRSD):
            # Cover the float-image-normalized-to-[0, 1] overlay rescaling branch
            x['image'] = x['image'] / 255.0
            dataset.plot(x)
            plt.close()
