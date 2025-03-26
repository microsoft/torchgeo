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
    @pytest.fixture(
        params=[
            ('cloud_s2', 'l1_cloud_s2', {}),
            ('cloud_s3', 'l1_cloud_s3', {'mode': 'binary'}),
            ('cloud_s3', 'l1_cloud_s3', {'mode': 'multi'}),
            ('eurosat_s1', 'l2_eurosat_s1s2', {}),
            ('eurosat_s2', 'l2_eurosat_s1s2', {}),
            ('bigearthnet_s1', 'l2_bigearthnet_s1s2', {}),
            ('bigearthnet_s2', 'l2_bigearthnet_s1s2', {}),
            ('lc100cls_s3', 'l2_lc100_s3', {'mode': 'static'}),
            ('lc100cls_s3', 'l2_lc100_s3', {'mode': 'time-series'}),
            ('lc100seg_s3', 'l2_lc100_s3', {'mode': 'static'}),
            ('lc100seg_s3', 'l2_lc100_s3', {'mode': 'time-series'}),
            ('dfc2020_s1', 'l2_dfc2020_s1s2', {}),
            ('dfc2020_s2', 'l2_dfc2020_s1s2', {}),
            ('flood_s1', 'l3_flood_s1', {'mode': 1}),
            ('flood_s1', 'l3_flood_s1', {'mode': 2}),
            ('lcz_s2', 'l3_lcz_s2', {}),
            ('biomass_s3', 'l3_biomass_s3', {'mode': 'static'}),
            ('biomass_s3', 'l3_biomass_s3', {'mode': 'time-series'}),
        ]
    )
    def dataset(self, request: SubRequest) -> CopernicusBench:
        dataset, directory, kwargs = request.param

        if dataset == 'lcz_s2':
            pytest.importorskip('h5py', minversion='3.8')

        root = os.path.join('tests', 'data', 'copernicus', directory)
        transforms = nn.Identity()
        return CopernicusBench(dataset, root, transforms=transforms, **kwargs)

    def test_getitem(self, dataset: CopernicusBench) -> None:
        x = dataset[0]
        assert isinstance(x['image'], torch.Tensor)
        if not dataset.name.startswith(('dfc2020', 'lcz')):
            assert isinstance(x['lat'], torch.Tensor)
            assert isinstance(x['lon'], torch.Tensor)
        if not dataset.name.startswith(('eurosat', 'dfc2020', 'lcz')):
            assert isinstance(x['time'], torch.Tensor)
        if 'label' in x:
            assert isinstance(x['label'], torch.Tensor)
        if 'mask' in x:
            assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: CopernicusBench) -> None:
        assert len(dataset) == 1

    def test_extract(self, dataset: CopernicusBench, tmp_path: Path) -> None:
        root = dataset.root
        if dataset.name == 'lcz_s2':
            file = dataset.filename.format(dataset.split)
        else:
            file = dataset.zipfile
        shutil.copyfile(os.path.join(root, file), tmp_path / file)
        CopernicusBench(dataset.name, tmp_path)

    def test_download(
        self, dataset: CopernicusBench, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        if dataset.name == 'lcz_s2':
            url = os.path.join(dataset.root, dataset.filename.format(dataset.split))
        else:
            url = os.path.join(dataset.root, dataset.zipfile)
        monkeypatch.setattr(dataset.dataset.__class__, 'url', url)
        CopernicusBench(dataset.name, tmp_path, download=True)

    def test_not_downloaded(self, dataset: CopernicusBench, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CopernicusBench(dataset.name, tmp_path)

    def test_plot(self, dataset: CopernicusBench) -> None:
        x = dataset[0]
        if 'label' in x:
            x['prediction'] = x['label']
        elif 'mask' in x:
            x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_not_rgb(self, dataset: CopernicusBench) -> None:
        all_bands = list(dataset.all_bands)
        rgb_bands = list(dataset.rgb_bands)
        for band in rgb_bands:
            all_bands.remove(band)

        if dataset.name.endswith('s1'):
            all_bands = ['VV']

        dataset = CopernicusBench(dataset.name, dataset.root, bands=all_bands)
        match = 'Dataset does not contain some of the RGB bands'
        with pytest.raises(RGBBandsMissingError, match=match):
            dataset.plot(dataset[0])
