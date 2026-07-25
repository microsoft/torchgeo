# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import CropHarvest, DatasetNotFoundError

pytest.importorskip('h5py', minversion='3.10')


class TestCropHarvest:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
    ) -> CropHarvest:
        split = request.param
        monkeypatch.setitem(
            CropHarvest.file_dict['features'], 'md5', 'ef6f4f00c0b3b50ed8380b0044928572'
        )
        monkeypatch.setitem(
            CropHarvest.file_dict['labels'], 'md5', 'f990f1bdfd9dc7efe99e94a9f511efde'
        )
        monkeypatch.setitem(
            CropHarvest.file_dict['features'],
            'url',
            os.path.join('tests', 'data', 'cropharvest', 'features.tar.gz'),
        )
        monkeypatch.setitem(
            CropHarvest.file_dict['labels'],
            'url',
            os.path.join('tests', 'data', 'cropharvest', 'labels.geojson'),
        )

        root = tmp_path
        transforms = nn.Identity()

        dataset = CropHarvest(
            root, split=split, transforms=transforms, download=True, checksum=True
        )
        return dataset

    def test_getitem(self, dataset: CropHarvest) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['array'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['array'].shape == (12, 18)

    def test_len(self, dataset: CropHarvest) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: CropHarvest, tmp_path: Path) -> None:
        CropHarvest(root=tmp_path, download=False)

    def test_downloaded_zipped(self, dataset: CropHarvest, tmp_path: Path) -> None:
        feature_path = os.path.join(tmp_path, 'features')
        shutil.rmtree(feature_path)
        CropHarvest(root=tmp_path, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CropHarvest(tmp_path)

    def test_plot(self, dataset: CropHarvest) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
