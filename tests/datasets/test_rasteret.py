# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pickle
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import torch
from pyproj import CRS
from torch import Tensor

from torchgeo.datasets import RasteretDataset

rasteret = pytest.importorskip('rasteret')

if TYPE_CHECKING:
    from rasteret.core.collection import Collection

# A Rasteret index generated from a real torchgeo Sentinel-2 patch (EPSG:32639,
# 10 m) by tests/data/rasteret/data.py. Loaded read-only here so no socket or
# raster read is needed; reads are stubbed per test.
WORKSPACE = 'tests/data/rasteret'


class TestRasteretDataset:
    """Tests for :class:`torchgeo.datasets.RasteretDataset`."""

    @pytest.fixture
    def collection(self) -> 'Collection':
        return rasteret.load(WORKSPACE, name='s2')

    def test_init_builds_index_from_footprints(self, collection: 'Collection') -> None:
        """The index and grid are derived from collection metadata, no raster open."""
        ds = RasteretDataset(collection, bands=['B04'])

        assert len(ds) == 2
        assert set(ds.index['id']) == {'patch_a', 'patch_b'}
        assert ds.bands == ('B04',)
        assert ds.crs == CRS.from_epsg(32639)
        assert ds.res == (10.0, 10.0)

    def test_init_defaults_to_all_bands(self, collection: 'Collection') -> None:
        """Omitting bands loads every band in the collection, like RasterDataset."""
        ds = RasteretDataset(collection)
        assert ds.bands == tuple(collection.bands)

    def test_init_custom_crs_and_res(self, collection: 'Collection') -> None:
        """CRS and resolution overrides are honored."""
        ds = RasteretDataset(
            collection, bands=['B04'], crs=CRS.from_epsg(4326), res=0.0001
        )
        assert ds.crs == CRS.from_epsg(4326)
        assert ds.res == (0.0001, 0.0001)

    def test_getitem(
        self, collection: 'Collection', monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """__getitem__ reads through collection.read_window onto the query grid."""
        captured: dict[str, object] = {}

        def read_window(**kwargs: object) -> npt.NDArray[np.uint16]:
            captured.update(kwargs)
            return np.ones((1, 16, 16), dtype=np.uint16)

        monkeypatch.setattr(collection, 'read_window', read_window)
        ds = RasteretDataset(collection, bands=['B04'])
        sample = ds[ds.bounds]

        assert sample['image'].shape == (1, 16, 16)
        assert sample['image'].dtype == torch.float32
        assert captured['bands'] == ['B04']
        assert captured['target_crs'] == 32639
        # index is ordered by (datetime, id), so record order is deterministic
        assert captured['record_ids'] == ['patch_a', 'patch_b']

    def test_getitem_no_match_raises(self, collection: 'Collection') -> None:
        """Queries that intersect no records raise IndexError."""
        ds = RasteretDataset(collection, bands=['B04'])
        _, _, t = ds.bounds
        with pytest.raises(IndexError, match='not found in dataset'):
            ds[1_000_000:1_000_010:10, 1_000_000:1_000_010:10, t]

    def test_time_series_passes_group_by(
        self, collection: 'Collection', monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """time_series=True stacks one timestep per record via group_by='id', matching
        RasterDataset(time_series=True)'s one-timestep-per-file stacking."""
        captured: dict[str, object] = {}

        def read_window(**kwargs: object) -> npt.NDArray[np.uint16]:
            captured.update(kwargs)
            return np.ones((2, 1, 16, 16), dtype=np.uint16)

        monkeypatch.setattr(collection, 'read_window', read_window)
        ds = RasteretDataset(collection, bands=['B04'], time_series=True)
        sample = ds[ds.bounds]

        assert captured['group_by'] == 'id'
        assert sample['image'].shape == (2, 1, 16, 16)

    def test_transforms_applied(
        self, collection: 'Collection', monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transforms callable is applied to each returned sample."""
        monkeypatch.setattr(
            collection, 'read_window', lambda **_: np.ones((1, 16, 16), dtype=np.uint16)
        )

        def double(sample: dict[str, Tensor]) -> dict[str, Tensor]:
            sample['image'] = sample['image'] * 2
            return sample

        ds = RasteretDataset(collection, bands=['B04'], transforms=double)
        assert torch.equal(ds[ds.bounds]['image'], torch.full((1, 16, 16), 2.0))

    def test_crs_is_fixed_after_construction(self, collection: 'Collection') -> None:
        """CRS is bound when the index is built, so reassigning it is rejected."""
        ds = RasteretDataset(collection, bands=['B04'])
        with pytest.raises(AttributeError, match='fixed at construction'):
            ds.crs = CRS.from_epsg(4326)

    def test_pickling(self, collection: 'Collection') -> None:
        """Dataset survives a pickle round-trip for multiprocessing support."""
        ds = RasteretDataset(collection, bands=['B04'])
        restored = pickle.loads(pickle.dumps(ds))

        assert len(restored.index) == len(ds.index)
        assert restored.bands == ds.bands
        assert restored.crs == ds.crs
