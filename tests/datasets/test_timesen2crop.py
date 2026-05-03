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

from torchgeo.datasets import DatasetNotFoundError, RGBBandsMissingError, TimeSen2Crop

TEST_TILES = ('33TUN', '2019_33UVP')
TEST_TILE_T = {'33TUN': 5, '2019_33UVP': 7}


class TestTimeSen2Crop:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> TimeSen2Crop:
        url = os.path.join('tests', 'data', 'timesen2crop', 'TimeSen2Crop.zip')
        monkeypatch.setattr(TimeSen2Crop, 'url', url)
        monkeypatch.setattr(TimeSen2Crop, 'md5', '63729536a368e643b5819d1cdf101a37')
        monkeypatch.setattr(TimeSen2Crop, 'valid_tiles', TEST_TILES)
        return TimeSen2Crop(
            root=tmp_path,
            tiles=TEST_TILES,
            transforms=nn.Identity(),
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: TimeSen2Crop) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].ndim == 2
        assert x['image'].shape[1] == len(TimeSen2Crop.all_bands)
        assert x['image'].shape[0] in TEST_TILE_T.values()
        assert x['mask'].shape == (x['image'].shape[0],)
        assert x['label'].ndim == 0
        assert 0 <= int(x['label']) < len(TimeSen2Crop.classes)
        assert x['mask'].dtype == torch.int64
        assert x['image'].dtype == torch.float32

    def test_len(self, dataset: TimeSen2Crop) -> None:
        # 2 tiles * 2 classes * 2 samples
        assert len(dataset) == 8

    def test_tile_dates(self, dataset: TimeSen2Crop) -> None:
        for tile, T in TEST_TILE_T.items():
            assert len(dataset.tile_dates[tile]) == T

    def test_bands_subset(self, dataset: TimeSen2Crop) -> None:
        ds = TimeSen2Crop(root=dataset.root, tiles=TEST_TILES, bands=('B4', 'B3', 'B2'))
        x = ds[0]
        assert x['image'].shape[1] == 3

    def test_tiles_subset(self, dataset: TimeSen2Crop) -> None:
        ds = TimeSen2Crop(root=dataset.root, tiles=('33TUN',))
        assert len(ds) == 4
        x = ds[0]
        assert x['image'].shape[0] == TEST_TILE_T['33TUN']

    def test_pad_to(self, dataset: TimeSen2Crop) -> None:
        max_T = max(TEST_TILE_T.values())
        ds = TimeSen2Crop(root=dataset.root, tiles=TEST_TILES, pad_to=max_T)
        for i in range(len(ds)):
            x = ds[i]
            assert x['image'].shape[0] == max_T
            assert x['mask'].shape[0] == max_T

        # The shorter tile must contain padding sentinels
        for i, (tile, _, _) in enumerate(ds.index):
            if tile == '33TUN':
                assert (ds[i]['mask'] == TimeSen2Crop.PADDING_VALUE).any()
                break

    def test_pad_to_too_small(self, dataset: TimeSen2Crop) -> None:
        with pytest.raises(AssertionError, match='pad_to'):
            TimeSen2Crop(root=dataset.root, tiles=TEST_TILES, pad_to=2)

    def test_invalid_band(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='Invalid band'):
            TimeSen2Crop(root=tmp_path, bands=('NOPE',))

    def test_invalid_tile(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='Invalid tile'):
            TimeSen2Crop(root=tmp_path, tiles=('NOPE',))

    def test_already_downloaded(self, dataset: TimeSen2Crop) -> None:
        TimeSen2Crop(root=dataset.root, tiles=TEST_TILES)

    def test_partial_cache(
        self, dataset: TimeSen2Crop, monkeypatch: MonkeyPatch
    ) -> None:
        # Delete one tile's cache; ctor should rebuild only that tile.
        cache_dir = os.path.join(dataset.root, TimeSen2Crop.cache_dirname)
        os.remove(os.path.join(cache_dir, '33TUN.npz'))
        monkeypatch.setattr(TimeSen2Crop, 'valid_tiles', TEST_TILES)
        TimeSen2Crop(root=dataset.root, tiles=TEST_TILES)

    def test_already_extracted(
        self, dataset: TimeSen2Crop, monkeypatch: MonkeyPatch
    ) -> None:
        # Wipe cache, leave the extracted Dataset/ in place; ctor should rebuild.
        shutil.rmtree(os.path.join(dataset.root, TimeSen2Crop.cache_dirname))
        monkeypatch.setattr(TimeSen2Crop, 'valid_tiles', TEST_TILES)
        TimeSen2Crop(root=dataset.root, tiles=TEST_TILES)

    def test_downloaded_zipped(
        self, dataset: TimeSen2Crop, monkeypatch: MonkeyPatch
    ) -> None:
        # Wipe cache and extracted dir, leave only the zip; ctor should re-extract.
        shutil.rmtree(os.path.join(dataset.root, TimeSen2Crop.cache_dirname))
        shutil.rmtree(os.path.join(dataset.root, TimeSen2Crop.extracted_dirname))
        monkeypatch.setattr(TimeSen2Crop, 'valid_tiles', TEST_TILES)
        TimeSen2Crop(root=dataset.root, tiles=TEST_TILES)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            TimeSen2Crop(tmp_path)

    def test_plot(self, dataset: TimeSen2Crop) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()
        dataset.plot(dataset[0], show_titles=False)
        plt.close()

    def test_plot_rgb_missing(self, dataset: TimeSen2Crop) -> None:
        ds = TimeSen2Crop(root=dataset.root, tiles=TEST_TILES, bands=('B5',))
        with pytest.raises(RGBBandsMissingError):
            ds.plot(ds[0])
