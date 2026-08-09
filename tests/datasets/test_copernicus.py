# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import io
import os
import shutil
import tarfile
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest
import requests
import torch
import torch.utils.data
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import (
    CopernicusBench,
    CopernicusEmbed,
    CopernicusPretrain,
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    UnionDataset,
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
            ('aq_no2_s5p', 'l3_airquality_s5p', {'mode': 'annual'}),
            ('aq_no2_s5p', 'l3_airquality_s5p', {'mode': 'seasonal'}),
            ('aq_o3_s5p', 'l3_airquality_s5p', {'mode': 'annual'}),
            ('aq_o3_s5p', 'l3_airquality_s5p', {'mode': 'seasonal'}),
        ]
    )
    def dataset(self, request: SubRequest) -> CopernicusBench:
        dataset, directory, kwargs = request.param

        if dataset == 'lcz_s2':
            pytest.importorskip('h5py', minversion='3.10')

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
        all_bands = list(dataset.dataset.all_bands)
        rgb_bands = list(dataset.dataset.rgb_bands)
        for band in rgb_bands:
            all_bands.remove(band)

        if dataset.name.endswith('s1'):
            all_bands = ['VV']
        elif dataset.name.endswith('s5p'):
            # single-band dataset
            return

        dataset = CopernicusBench(dataset.name, dataset.root, bands=all_bands)
        match = 'Dataset does not contain some of the RGB bands'
        with pytest.raises(RGBBandsMissingError, match=match):
            dataset.plot(dataset[0])


def create_shard(path: Path, keys: list[str], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = [
            's1_grd',
            's2_toa',
            's3_olci',
            's5p_co',
            's5p_no2',
            's5p_o3',
            's5p_so2',
            'dem',
        ]
    with tarfile.open(path, 'w') as tar:
        for key in keys:
            for field in fields:
                buffer = io.BytesIO()
                torch.save(torch.rand(2, 2, 2), buffer)
                data = buffer.getvalue()
                info = tarfile.TarInfo(f'{key}.{field}.pth')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


class WorkerInfo(NamedTuple):
    id: int
    num_workers: int


class TestCopernicusPretrain:
    @pytest.fixture
    def urls(self) -> str:
        root = os.path.join('tests', 'data', 'copernicus', 'pretrain')
        shards = 'example-{000000..000000}.tar'
        return os.path.join(root, shards)

    @pytest.fixture
    def dataset(self, urls: str) -> CopernicusPretrain:
        return CopernicusPretrain(urls, shardshuffle=False)

    def test_getitem(self, dataset: CopernicusPretrain) -> None:
        x = next(iter(dataset))
        # Check the types of the tensors
        assert isinstance(x['s1_grd.pth'], torch.Tensor)
        assert isinstance(x['s2_toa.pth'], torch.Tensor)
        assert isinstance(x['s3_olci.pth'], torch.Tensor)
        assert isinstance(x['s5p_co.pth'], torch.Tensor)
        assert isinstance(x['s5p_no2.pth'], torch.Tensor)
        assert isinstance(x['s5p_o3.pth'], torch.Tensor)
        assert isinstance(x['s5p_so2.pth'], torch.Tensor)
        assert isinstance(x['dem.pth'], torch.Tensor)
        # Check the shapes of the tensors
        assert x['s1_grd.pth'].shape == (2, 264, 264)
        assert x['s2_toa.pth'].shape == (13, 264, 264)
        assert x['s3_olci.pth'].shape == (21, 96, 96)
        assert x['s5p_co.pth'].shape == (1, 28, 28)
        assert x['s5p_no2.pth'].shape == (1, 28, 28)
        assert x['s5p_o3.pth'].shape == (1, 28, 28)
        assert x['s5p_so2.pth'].shape == (1, 28, 28)
        assert x['dem.pth'].shape == (960, 960)

    def test_plot(self, dataset: CopernicusPretrain) -> None:
        x = next(iter(dataset))
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_expand_urls(self, urls: str) -> None:
        assert len(CopernicusPretrain('example-{000000..000009}.tar').urls) == 10
        assert CopernicusPretrain('example-000000.tar').urls == ['example-000000.tar']
        assert len(CopernicusPretrain([urls, urls]).urls) == 2

    def test_shardshuffle(self, urls: str) -> None:
        dataset = CopernicusPretrain(urls, shardshuffle=True)
        assert len(list(dataset)) == 1

    def test_resampled(self, urls: str) -> None:
        dataset = CopernicusPretrain(urls, resampled=True)
        it = iter(dataset)
        for _ in range(2):
            assert isinstance(next(it)['s1_grd.pth'], torch.Tensor)

    def test_shuffle_buffer(self, urls: str) -> None:
        dataset = CopernicusPretrain([urls, urls], shuffle_buffer=1)
        assert len(list(dataset)) == 2

    def test_remote_shards(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        path = tmp_path / 'example-000000.tar'
        create_shard(path, keys=['a', 'b'])

        class Response:
            raw: io.BufferedReader

            def raise_for_status(self) -> None:
                pass

        with open(path, 'rb') as f:
            Response.raw = f
            monkeypatch.setattr(requests, 'get', lambda url, **kwargs: Response())
            dataset = CopernicusPretrain('https://example.com/example-000000.tar')
            assert len(list(dataset)) == 2

    def test_worker_split(self, monkeypatch: MonkeyPatch, urls: str) -> None:
        monkeypatch.setattr(
            torch.utils.data, 'get_worker_info', lambda: WorkerInfo(1, 2)
        )
        dataset = CopernicusPretrain([urls, urls])
        assert len(list(dataset)) == 1

    def test_distributed_split(self, monkeypatch: MonkeyPatch, urls: str) -> None:
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
        monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
        dataset = CopernicusPretrain([urls, urls])
        assert len(list(dataset)) == 1

    def test_missing_modalities(self, tmp_path: Path) -> None:
        path = tmp_path / 'example-000000.tar'
        create_shard(path, keys=['a'], fields=['s1_grd'])
        dataset = CopernicusPretrain(str(path))
        assert list(dataset) == []


class TestCopernicusEmbed:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch) -> CopernicusEmbed:
        paths = os.path.join('tests', 'data', 'copernicus', 'embed')
        monkeypatch.setattr(
            CopernicusEmbed, 'url', os.path.join(paths, 'embed_map_310k.tif')
        )
        transforms = nn.Identity()
        return CopernicusEmbed(paths, transforms=transforms)

    def test_len(self, dataset: CopernicusEmbed) -> None:
        assert len(dataset) == 1

    def test_getitem(self, dataset: CopernicusEmbed) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_and(self, dataset: CopernicusEmbed) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: CopernicusEmbed) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: CopernicusEmbed) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_download(self, dataset: CopernicusEmbed, tmp_path: Path) -> None:
        CopernicusEmbed(tmp_path, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CopernicusEmbed(tmp_path)

    def test_invalid_index(self, dataset: CopernicusEmbed) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
