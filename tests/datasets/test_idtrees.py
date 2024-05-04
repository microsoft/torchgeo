# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import glob
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import DatasetNotFoundError, IDTReeS

from .utils import importandskip


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestIDTReeS:
    @pytest.fixture(params=zip(['train', 'test', 'test'], ['task1', 'task1', 'task2']))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> IDTReeS:
        monkeypatch.setattr(torchgeo.datasets.idtrees, 'download_url', download_url)
        data_dir = os.path.join('tests', 'data', 'idtrees')
        metadata = {
            'train': {
                'url': os.path.join(data_dir, 'IDTREES_competition_train_v2.zip'),
                'md5': '5ddfa76240b4bb6b4a7861d1d31c299c',
                'filename': 'IDTREES_competition_train_v2.zip',
            },
            'test': {
                'url': os.path.join(data_dir, 'IDTREES_competition_test_v2.zip'),
                'md5': 'b108931c84a70f2a38a8234290131c9b',
                'filename': 'IDTREES_competition_test_v2.zip',
            },
        }
        split, task = request.param
        monkeypatch.setattr(IDTReeS, 'metadata', metadata)
        root = str(tmp_path)
        transforms = nn.Identity()
        return IDTReeS(root, split, task, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: IDTReeS) -> None:
        pytest.importorskip('laspy', minversion='2')
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['chm'], torch.Tensor)
        assert isinstance(x['hsi'], torch.Tensor)
        assert isinstance(x['las'], torch.Tensor)
        assert x['image'].shape == (3, 200, 200)
        assert x['chm'].shape == (1, 200, 200)
        assert x['hsi'].shape == (369, 200, 200)
        assert x['las'].ndim == 2
        assert x['las'].shape[0] == 3

        if 'label' in x:
            assert isinstance(x['label'], torch.Tensor)
        if 'boxes' in x:
            assert isinstance(x['boxes'], torch.Tensor)
            if x['boxes'].ndim != 1:
                assert x['boxes'].ndim == 2
                assert x['boxes'].shape[-1] == 4

    def test_len(self, dataset: IDTReeS) -> None:
        assert len(dataset) == 3

    def test_already_downloaded(self, dataset: IDTReeS) -> None:
        IDTReeS(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        pytest.importorskip('laspy', minversion='2')
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            IDTReeS(str(tmp_path))

    def test_not_extracted(self, tmp_path: Path) -> None:
        pytest.importorskip('laspy', minversion='2')
        pathname = os.path.join('tests', 'data', 'idtrees', '*.zip')
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        IDTReeS(root)

    def test_missing_module(self, dataset: IDTReeS) -> None:
        importandskip('laspy')
        match='laspy is not installed and is required to use this dataset'
        with pytest.raises(ImportError, match=match):
            IDTReeS(dataset.root, dataset.split, dataset.task)

        importandskip('pyvista')
        match='pyvista is not installed and is required to plot point cloud'
        with pytest.raises(ImportError, match=match):
            dataset.plot_las(0)

    def test_plot(self, dataset: IDTReeS) -> None:
        pytest.importorskip('laspy', minversion='2')
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        if 'boxes' in x:
            x['prediction_boxes'] = x['boxes']
            dataset.plot(x, show_titles=True)
            plt.close()
        if 'label' in x:
            x['prediction_label'] = x['label']
            dataset.plot(x, show_titles=False)
            plt.close()

    def test_plot_las(self, dataset: IDTReeS) -> None:
        pytest.importorskip('laspy', minversion='2')
        pyvista = pytest.importorskip('pyvista', minversion='0.34.2')
        pyvista.OFF_SCREEN = True

        # Test point cloud without colors
        point_cloud = dataset.plot_las(index=0)
        pyvista.plot(point_cloud, scalars=point_cloud.points, cpos='yz', cmap='viridis')
