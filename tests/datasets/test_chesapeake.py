# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import itertools
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import (
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
        sha256s = {2018: ''}
        monkeypatch.setattr(ChesapeakeDC, 'sha256s', sha256s)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        transforms = nn.Identity()
        return ChesapeakeDC(tmp_path, transforms=transforms, download=True)

    def test_getitem(self, dataset: ChesapeakeDC) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
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
            ChesapeakeDC(tmp_path)

    def test_plot(self, dataset: ChesapeakeDC) -> None:
        index = dataset.bounds
        x = dataset[index]
        dataset.plot(x, suptitle='Test')
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_index(self, dataset: ChesapeakeDC) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]


class TestChesapeakeCVPR:
    @pytest.fixture(
        params=[
            params
            for params in itertools.product(
                [['de'], ['de', 'md']],
                [['test'], ['test', 'train']],
                [[], ['naip-new'], ['naip-new', 'landsat-leaf-on']],
                [
                    [],
                    ['nlcd', 'lc', 'buildings'],
                    ['prior_from_cooccurrences_101_31_no_osm_no_buildings'],
                ],
            )
            if params[2] or params[3]
        ]
    )
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> ChesapeakeCVPR:
        state, split, image, mask = request.param
        data_splits = [f'{s}-{sp}' for s in state for sp in split]
        data_layers = image + mask
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
            {
                'base': ('de_1m_2013_extended-debuffered-test_tiles',),
                'prior_extension': (
                    'de_1m_2013_extended-debuffered-test_tiles/m_3807504_ne_18_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif',
                ),
            },
        )
        root = tmp_path
        transforms = nn.Identity()
        return ChesapeakeCVPR(
            root,
            splits=data_splits,
            layers=data_layers,
            transforms=transforms,
            download=True,
        )

    def test_getitem(self, dataset: ChesapeakeCVPR) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x.get('mask', x.get('image')), torch.Tensor)

    def test_len(self, dataset: ChesapeakeCVPR) -> None:
        assert len(dataset) > 0

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
            ChesapeakeCVPR(tmp_path)

    def test_base_only_without_prior_extension(self, tmp_path: Path) -> None:
        shutil.copy(
            os.path.join(
                'tests', 'data', 'chesapeake', 'cvpr', 'cvpr_chesapeake_landcover.zip'
            ),
            tmp_path,
        )
        ChesapeakeCVPR(tmp_path, splits=['de-test'], layers=['naip-new', 'lc'])

    def test_prior_extension_missing(self, tmp_path: Path) -> None:
        shutil.copy(
            os.path.join(
                'tests', 'data', 'chesapeake', 'cvpr', 'cvpr_chesapeake_landcover.zip'
            ),
            tmp_path,
        )
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ChesapeakeCVPR(
                tmp_path,
                splits=['de-test'],
                layers=['naip-new', ChesapeakeCVPR.prior_layer],
            )

    def test_out_of_bounds_index(self, dataset: ChesapeakeCVPR) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_plot(self, dataset: ChesapeakeCVPR) -> None:
        x = dataset[dataset.bounds].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        if 'mask' in x:
            if x['mask'].ndim == 2:
                x['prediction'] = x['mask'].clone()
            else:
                x['prediction'] = x['mask'][0, :, :].clone()
            dataset.plot(x)
            plt.close()

    def test_partially_out_of_raster_query(self, dataset: ChesapeakeCVPR) -> None:
        # Regression test for https://github.com/torchgeo/torchgeo/issues/3678
        # Constructs a query whose right half lies outside the tile's raster
        # footprint (after EPSG:3857 -> UTM reprojection). Before the fix the
        # dataset returned a clipped tensor, which broke downstream transforms
        # like K.CenterCrop. After the fix the returned tensor must match the
        # requested patch shape, with the out-of-raster region zero-filled.
        x, y, t = dataset.bounds
        xshift = (x.stop - x.start) / 2
        shifted_x = slice(x.start + xshift, x.stop + xshift, x.step)
        sample = dataset[shifted_x, y, t]
        ref = dataset[dataset.bounds]
        # The shifted-right half of the query is outside the raster, so at
        # least one column on the right should be zero-filled.
        if 'image' in sample:
            assert sample['image'].shape == ref['image'].shape
            assert torch.all(sample['image'][..., -1] == 0)
        if 'mask' in sample:
            assert sample['mask'].shape == ref['mask'].shape
            assert torch.all(sample['mask'][..., -1] == 0)
