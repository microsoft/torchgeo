# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pickle
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, create_autospec

import geopandas as gpd
import pandas as pd
import pytest
import shapely
import torch
from pyproj import CRS

pa = pytest.importorskip('pyarrow')
pads = pytest.importorskip('pyarrow.dataset')
pytest.importorskip('rasteret')

if TYPE_CHECKING:
    from rasteret import Collection

Collection = import_module('rasteret').Collection


class _DummyRasteretGeoDataset:
    """Minimal stand-in for rasteret.integrations.torchgeo.RasteretGeoDataset."""

    def __init__(self) -> None:
        interval_index = pd.IntervalIndex.from_tuples(
            [(datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))],
            closed='both',
            name='datetime',
        )
        geometry = [shapely.box(399960, 5390220, 400600, 5390860)]
        self.index = gpd.GeoDataFrame(
            {'rid': [0]},
            index=interval_index,
            geometry=geometry,
            crs=CRS.from_epsg(32632),
        )
        self._res = (10.0, 10.0)
        self.closed = False

    @property
    def crs(self) -> CRS:
        return CRS.from_user_input(self.index.crs)

    @property
    def res(self) -> tuple[float, float]:
        return self._res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        if isinstance(new_res, int | float):
            new_res = (float(new_res), float(new_res))
        self._res = new_res

    def __getitem__(self, _index: Any) -> dict[str, torch.Tensor]:
        return {
            'image': torch.ones((3, 16, 16), dtype=torch.float32),
            'bounds': torch.zeros(9, dtype=torch.float64),
            'transform': torch.zeros(9, dtype=torch.float64),
        }

    def close(self) -> None:
        self.closed = True


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


def _make_real_collection() -> Collection:
    table = pa.table(
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
    return Collection(dataset=pads.dataset(table), data_source='test-rasteret')


class TestRasteretDataset:
    """Tests for :class:`torchgeo.datasets.RasteretDataset`."""

    @pytest.fixture
    def delegate(self) -> _DummyRasteretGeoDataset:
        return _DummyRasteretGeoDataset()

    @pytest.fixture
    def collection(self, delegate: _DummyRasteretGeoDataset) -> MagicMock:
        collection = create_autospec(Collection, instance=True)
        collection.to_torchgeo_dataset.return_value = delegate
        return collection

    @pytest.fixture
    def real_collection(self) -> Collection:
        return _make_real_collection()

    def test_init(
        self, collection: MagicMock, delegate: _DummyRasteretGeoDataset
    ) -> None:
        """RasteretDataset delegates creation to collection.to_torchgeo_dataset."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04', 'B03', 'B02'])

        call = collection.to_torchgeo_dataset.call_args
        assert call is not None
        assert call.kwargs['bands'] == ['B04', 'B03', 'B02']
        assert call.kwargs['target_crs'] is None
        assert len(ds.index) == 1
        assert ds.bands == ('B04', 'B03', 'B02')
        assert ds.crs == CRS.from_epsg(32632)
        assert ds.res == delegate.res

    def test_init_custom_crs_res(self, collection: MagicMock) -> None:
        """CRS and resolution overrides are forwarded correctly."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(
            collection=collection, bands=['B04'], crs=CRS.from_epsg(4326), res=0.0001
        )

        call = collection.to_torchgeo_dataset.call_args
        assert call is not None
        assert call.kwargs['target_crs'] == 4326
        assert ds.res == (0.0001, 0.0001)

    def test_init_non_epsg_crs(self, collection: MagicMock) -> None:
        """CRS overrides must be EPSG-resolvable."""
        from torchgeo.datasets import RasteretDataset

        with pytest.raises(ValueError, match='EPSG'):
            RasteretDataset(collection=collection, bands=['B04'], crs=_non_epsg_crs())

    def test_init_requires_collection_adapter(self) -> None:
        """Collection must implement to_torchgeo_dataset."""
        from torchgeo.datasets import RasteretDataset

        with pytest.raises(TypeError, match='to_torchgeo_dataset'):
            RasteretDataset(collection=object(), bands=['B04'])

    def test_init_requires_callable_collection_adapter(self) -> None:
        """Collection adapter attribute must be callable."""
        from torchgeo.datasets import RasteretDataset

        class _NotCallableAdapter:
            to_torchgeo_dataset = 'not-callable'

        with pytest.raises(TypeError, match='to_torchgeo_dataset'):
            RasteretDataset(collection=_NotCallableAdapter(), bands=['B04'])

    def test_init_requires_bands(self, collection: MagicMock) -> None:
        """At least one band is required."""
        from torchgeo.datasets import RasteretDataset

        with pytest.raises(ValueError, match='At least one band'):
            RasteretDataset(collection=collection, bands=[])

    def test_getitem(self, collection: MagicMock) -> None:
        """__getitem__ delegates to Rasteret's dataset implementation."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04', 'B03', 'B02'])
        sample = ds[ds.bounds]

        assert 'image' in sample
        assert sample['image'].shape == (3, 16, 16)

    def test_res_setter(self, collection: MagicMock) -> None:
        """Resolution changes stay delegated to the Rasteret dataset."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        ds.res = 20.0

        assert ds.res == (20.0, 20.0)
        assert ds._delegate.res == (20.0, 20.0)

    def test_crs_setter_rejected(self, collection: MagicMock) -> None:
        """Post-init CRS mutation is rejected because Rasteret binds target CRS."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])

        with pytest.raises(AttributeError, match='fixed after construction'):
            ds.crs = CRS.from_epsg(4326)

    def test_dtype_property(self, collection: MagicMock) -> None:
        """dtype defaults correctly for image and mask styles."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        assert ds.dtype == torch.float32
        ds.is_image = False
        assert ds.dtype == torch.long

    def test_pickling(self, collection: MagicMock) -> None:
        """Dataset survives pickle round-trip for multiprocessing support."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        restored = pickle.loads(pickle.dumps(ds))

        assert len(restored.index) == len(ds.index)
        assert restored.bands == ds.bands
        sample = restored[restored.bounds]
        assert sample['image'].shape == (3, 16, 16)

    def test_close(self, collection: MagicMock) -> None:
        """close forwards to delegate close when available."""
        from torchgeo.datasets import RasteretDataset

        ds = RasteretDataset(collection=collection, bands=['B04'])
        ds.close()

        assert ds._delegate.closed is True
