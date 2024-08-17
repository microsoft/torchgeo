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
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    ChesapeakeCVPR,
    ChesapeakeDC,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestChesapeakeDC:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ChesapeakeDC:
        url = os.path.join(
            'tests',
            'data',
            'chesapeake',
            'lulc',
            '{state}_lulc_{year}_2022-Edition.zip',
        )
        monkeypatch.setattr(ChesapeakeDC, 'url', url)
        md5s = {2018: '35c644f13ccdb1baf62adf85cb8c7e48'}
        monkeypatch.setattr(ChesapeakeDC, 'md5s', md5s)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        transforms = nn.Identity()
        return ChesapeakeDC(
            tmp_path, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: ChesapeakeDC) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: ChesapeakeDC) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: ChesapeakeDC) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: ChesapeakeDC) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: ChesapeakeDC) -> None:
        ChesapeakeDC(dataset.paths, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        url = os.path.join(
            'tests', 'data', 'chesapeake', 'lulc', 'dc_lulc_2018_2022-Edition.zip'
        )
        shutil.copy(url, tmp_path)
        ChesapeakeDC(tmp_path)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ChesapeakeDC(tmp_path, checksum=True)

    def test_plot(self, dataset: ChesapeakeDC) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: ChesapeakeDC) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]


class TestChesapeakeCVPR:
    @pytest.fixture(
        params=[
            ('naip-new', 'naip-old', 'nlcd'),
            ('landsat-leaf-on', 'landsat-leaf-off', 'lc'),
            ('naip-new', 'landsat-leaf-on', 'lc', 'nlcd', 'buildings'),
            ('naip-new', 'prior_from_cooccurrences_101_31_no_osm_no_buildings'),
        ]
    )
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> ChesapeakeCVPR:
        monkeypatch.setattr(
            ChesapeakeCVPR,
            'md5s',
            {
                'base': '882d18b1f15ea4498bf54e674aecd5d4',
                'prior_extension': '677446c486f3145787938b14ee3da13f',
            },
        )
        monkeypatch.setattr(
            ChesapeakeCVPR,
            'urls',
            {
                'base': os.path.join(
                    'tests',
                    'data',
                    'chesapeake',
                    'cvpr',
                    'cvpr_chesapeake_landcover.zip',
                ),
                'prior_extension': os.path.join(
                    'tests',
                    'data',
                    'chesapeake',
                    'cvpr',
                    'cvpr_chesapeake_landcover_prior_extension.zip',
                ),
            },
        )
        monkeypatch.setattr(
            ChesapeakeCVPR,
            '_files',
            ['de_1m_2013_extended-debuffered-test_tiles', 'spatial_index.geojson'],
        )
        root = tmp_path
        transforms = nn.Identity()
        return ChesapeakeCVPR(
            root,
            splits=['de-test'],
            layers=request.param,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: ChesapeakeCVPR) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: ChesapeakeCVPR) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: ChesapeakeCVPR) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: ChesapeakeCVPR) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: ChesapeakeCVPR) -> None:
        ChesapeakeCVPR(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        root = tmp_path
        shutil.copy(
            os.path.join(
                'tests', 'data', 'chesapeake', 'cvpr', 'cvpr_chesapeake_landcover.zip'
            ),
            root,
        )
        shutil.copy(
            os.path.join(
                'tests',
                'data',
                'chesapeake',
                'cvpr',
                'cvpr_chesapeake_landcover_prior_extension.zip',
            ),
            root,
        )
        ChesapeakeCVPR(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ChesapeakeCVPR(tmp_path, checksum=True)

    def test_out_of_bounds_query(self, dataset: ChesapeakeCVPR) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_multiple_hits_query(self, dataset: ChesapeakeCVPR) -> None:
        ds = ChesapeakeCVPR(
            root=dataset.root, splits=['de-train', 'de-test'], layers=dataset.layers
        )
        with pytest.raises(
            IndexError, match='query: .* spans multiple tiles which is not valid'
        ):
            ds[dataset.bounds]

    def test_plot(self, dataset: ChesapeakeCVPR) -> None:
        x = dataset[dataset.bounds].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'][:, :, 0].clone().unsqueeze(2)
        dataset.plot(x)
        plt.close()
