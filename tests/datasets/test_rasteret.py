# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pickle
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from pyproj import CRS

pa = pytest.importorskip('pyarrow')
pads = pytest.importorskip('pyarrow.dataset')
pytest.importorskip('rasteret')

if TYPE_CHECKING:
    from rasteret import Collection

Collection = import_module('rasteret').Collection


def _non_epsg_crs() -> CRS:
    """Create a valid CRS object that is not EPSG-resolvable."""
    return CRS.from_wkt(
        'ENGCRS["foo",EDATUM["Unknown"],CS[Cartesian,2],'
        'AXIS["x",east,ORDER[1]],AXIS["y",north,ORDER[2]],'
        'LENGTHUNIT["metre",1]]'
    )


def _band_metadata(
    *, xmin: float, ymax: float, res: float = 10.0, width: int = 2, height: int = 2
) -> dict[str, Any]:
    return {
        'transform': [res, 0.0, xmin, 0.0, -res, ymax],
        'image_width': width,
        'image_height': height,
    }


def _real_table() -> Any:
    return pa.table(
        {
            'id': ['first', 'second'],
            'datetime': [
                datetime(2024, 6, 1, tzinfo=UTC),
                datetime(2024, 6, 2, tzinfo=UTC),
            ],
            'assets': [
                {'B04': {'href': 'memory://first.tif'}},
                {'B04': {'href': 'memory://second.tif'}},
            ],
            'proj:epsg': [32632, 32632],
            'B04_metadata': [
                _band_metadata(xmin=0.0, ymax=20.0),
                _band_metadata(xmin=10.0, ymax=20.0),
            ],
        }
    )


def _make_real_collection() -> Collection:
    return Collection(dataset=pads.dataset(_real_table()), data_source='test-rasteret')


def _make_file_backed_collection(path: Any) -> Collection:
    """Persist to parquet so the dataset (and thus the Collection) is picklable."""
    import pyarrow.parquet as pq

    pq.write_table(_real_table(), str(path))
    return Collection(dataset=pads.dataset(str(path)), data_source='test-rasteret')


class TestRasteretDataset:
    """Tests for :class:`torchgeo.datasets.RasteretDataset`."""

    @pytest.fixture
    def collection(self) -> Collection:
        return _make_real_collection()

    def test_init_builds_index_from_footprints(self, collection: Collection) -> None:
        """The index and grid are derived from collection metadata, no raster open."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])

        assert len(ds.index) == 2
        assert list(ds.index['id']) == ['first', 'second']
        assert ds.bands == ('B04',)
        assert ds.crs == CRS.from_epsg(32632)
        assert ds.res == (10.0, 10.0)

    def test_init_custom_crs_res(self, collection: Collection) -> None:
        """CRS and resolution overrides are honored."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(
            collection=collection, bands=['B04'], crs=CRS.from_epsg(4326), res=0.0001
        )

        assert ds.crs == CRS.from_epsg(4326)
        assert ds.res == (0.0001, 0.0001)

    def test_init_non_epsg_crs(self, collection: Collection) -> None:
        """CRS overrides must be EPSG-resolvable."""
        from torchgeo.datasets import RasteretDataset

        with pytest.raises(ValueError, match='EPSG'):
            RasteretDataset(collection=collection, bands=['B04'], crs=_non_epsg_crs())

    def test_init_requires_collection_boundary(self) -> None:
        """Collection must expose the Rasteret integration boundary."""
        from torchgeo.datasets import RasteretDataset

        with pytest.raises(TypeError, match='footprints'):
            RasteretDataset(collection=object(), bands=['B04'])

    def test_init_requires_bands(self, collection: Collection) -> None:
        """At least one band is required."""
        from torchgeo.datasets import RasteretDataset

        with pytest.raises(ValueError, match='At least one band'):
            RasteretDataset(collection=collection, bands=[])

    def test_len_and_files(self, collection: Collection) -> None:
        """len reflects the index; files lists the collection's COG hrefs."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        assert len(ds) == 2
        assert ds.files == ['memory://first.tif', 'memory://second.tif']

    def test_getitem(
        self, collection: Collection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """__getitem__ reads through collection.read_window onto the query grid."""
        from torchgeo.datasets import RasteretDataset

        captured: dict[str, Any] = {}

        def fake_read_window(**kwargs: Any) -> np.ndarray:
            captured.update(kwargs)
            return np.ones((1, 16, 16), dtype=np.uint16)

        monkeypatch.setattr(collection, 'read_window', fake_read_window)
        ds = RasteretDataset(collection=collection, bands=['B04'])
        sample = ds[ds.bounds]

        assert sample['image'].shape == (1, 16, 16)
        assert sample['image'].dtype == torch.float32
        assert 'bounds' in sample and 'transform' in sample
        assert captured['bands'] == ['B04']
        assert captured['target_crs'] == 32632
        assert sorted(captured['record_ids']) == ['first', 'second']

    def test_getitem_no_match_raises(self, collection: Collection) -> None:
        """Queries that intersect no records raise IndexError."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        _, _, t = ds.bounds
        with pytest.raises(IndexError, match='not found in dataset'):
            ds[1_000_000:1_000_010:10, 1_000_000:1_000_010:10, t]

    def test_time_series_passes_group_by(
        self, collection: Collection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """time_series=True groups the read by datetime."""
        from torchgeo.datasets import RasteretDataset

        captured: dict[str, Any] = {}

        def fake_read_window(**kwargs: Any) -> np.ndarray:
            captured.update(kwargs)
            return np.ones((2, 1, 16, 16), dtype=np.uint16)

        monkeypatch.setattr(collection, 'read_window', fake_read_window)
        ds = RasteretDataset(collection=collection, bands=['B04'], time_series=True)
        sample = ds[ds.bounds]

        assert captured['group_by'] == 'datetime'
        assert sample['image'].shape == (2, 1, 16, 16)

    def test_res_setter(self, collection: Collection) -> None:
        """Resolution can be overridden post-init."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        ds.res = 20.0
        assert ds.res == (20.0, 20.0)

    def test_crs_setter_rejected(self, collection: Collection) -> None:
        """Post-init CRS mutation is rejected because Rasteret binds target CRS."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        with pytest.raises(AttributeError, match='fixed after construction'):
            ds.crs = CRS.from_epsg(4326)

    def test_dtype_property(self, collection: Collection) -> None:
        """dtype defaults correctly for image and mask styles."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        assert ds.dtype == torch.float32
        ds.is_image = False
        assert ds.dtype == torch.long

    def test_pickling(self, tmp_path: Any) -> None:
        """Dataset survives a pickle round-trip for multiprocessing support."""
        from torchgeo.datasets import RasteretDataset

        collection = _make_file_backed_collection(tmp_path / 'collection.parquet')
        ds = RasteretDataset(collection=collection, bands=['B04'])
        restored = pickle.loads(pickle.dumps(ds))

        assert len(restored.index) == len(ds.index)
        assert restored.bands == ds.bands
        assert restored.crs == ds.crs

    def test_close(
        self, collection: Collection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """close is a safe no-op and forwards to the collection when supported."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        ds.close()  # collection exposes no close(): must not raise

        closed = {'value': False}
        monkeypatch.setattr(
            collection,
            'close',
            lambda: closed.__setitem__('value', True),
            raising=False,
        )
        ds.close()
        assert closed['value'] is True
