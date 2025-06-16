# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Sequence
from datetime import datetime
from math import floor, isclose
from typing import Any

import pandas as pd
import pytest
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import Geometry, Polygon
from torch import Generator

from torchgeo.datasets import (
    BoundingBox,
    GeoDataset,
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)

MINT = datetime(2025, 4, 24)
MAXT = datetime(2025, 4, 25)


def total_area(dataset: GeoDataset) -> float:
    area: float = dataset.index.geometry.area.sum()
    return area


def no_overlap(ds1: GeoDataset, ds2: GeoDataset) -> bool:
    try:
        ds = ds1 & ds2
    except RuntimeError:
        return True
    else:
        return isclose(total_area(ds), 0)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self,
        index: pd.IntervalIndex | None = None,
        geometry: Sequence[Geometry] = [shapely.box(0, 0, 1, 1)],
    ) -> None:
        if index is None:
            intervals = [(MINT, MAXT)] * len(geometry)
            index = pd.IntervalIndex.from_tuples(
                intervals, closed='both', name='datetime'
            )
        crs = CRS.from_epsg(3005)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)
        self.res = (1, 1)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        return {'index': query}


@pytest.mark.parametrize(
    'lengths,expected_lengths',
    [
        # List of lengths
        ([2, 1, 1], [2, 1, 1]),
        # List of fractions (with remainder)
        ([1 / 3, 1 / 3, 1 / 3], [2, 1, 1]),
    ],
)
def test_random_bbox_assignment(
    lengths: Sequence[int | float], expected_lengths: Sequence[int]
) -> None:
    geometry = [
        shapely.box(0, 0, 1, 1),
        shapely.box(1, 0, 2, 1),
        shapely.box(2, 0, 3, 1),
        shapely.box(3, 0, 4, 1),
    ]
    ds = CustomGeoDataset(geometry=geometry)

    train_ds, val_ds, test_ds = random_bbox_assignment(ds, lengths)

    # Check datasets lengths
    assert len(train_ds) == expected_lengths[0]
    assert len(val_ds) == expected_lengths[1]
    assert len(test_ds) == expected_lengths[2]

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds

    # Test __getitem__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)


def test_random_bbox_assignment_invalid_inputs() -> None:
    with pytest.raises(
        ValueError,
        match="Sum of input lengths must equal 1 or the length of dataset's index.",
    ):
        random_bbox_assignment(CustomGeoDataset(), lengths=[2, 2, 1])
    with pytest.raises(
        ValueError, match='All items in input lengths must be greater than 0.'
    ):
        random_bbox_assignment(CustomGeoDataset(), lengths=[1 / 2, 3 / 4, -1 / 4])


def test_random_bbox_splitting() -> None:
    geometry = [
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))),
        Polygon(((2, 0), (1, 0), (1, 1), (2, 1), (2, 0))),
        Polygon(((2, 2), (1, 2), (1, 1), (2, 1), (2, 2))),
        Polygon(((0, 2), (1, 2), (1, 1), (0, 1), (0, 2))),
    ]
    ds = CustomGeoDataset(geometry=geometry)

    ds_area = total_area(ds)

    train_ds, val_ds, test_ds = random_bbox_splitting(
        ds, fractions=[5 / 8, 2 / 8, 1 / 8], generator=Generator().manual_seed(5)
    )
    train_ds_area = total_area(train_ds)
    val_ds_area = total_area(val_ds)
    test_ds_area = total_area(test_ds)

    # Check datasets areas
    assert isclose(train_ds_area, ds_area * 5 / 8)
    assert isclose(val_ds_area, ds_area * 2 / 8)
    assert isclose(test_ds_area, ds_area * 1 / 8)

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds
    assert isclose(total_area(train_ds | val_ds | test_ds), ds_area)

    # Test __getitem__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)

    # Test invalid input fractions
    with pytest.raises(ValueError, match='Sum of input fractions must equal 1.'):
        random_bbox_splitting(ds, fractions=[1 / 2, 1 / 3, 1 / 4])
    with pytest.raises(
        ValueError, match='All items in input fractions must be greater than 0.'
    ):
        random_bbox_splitting(ds, fractions=[1 / 2, 3 / 4, -1 / 4])


def test_random_grid_cell_assignment() -> None:
    geometry = [shapely.box(0, 0, 12, 12), shapely.box(12, 0, 24, 12)]
    ds = CustomGeoDataset(geometry=geometry)

    train_ds, val_ds, test_ds = random_grid_cell_assignment(
        ds, fractions=[1 / 2, 1 / 4, 1 / 4], grid_size=5
    )

    # Check datasets lengths
    assert len(train_ds) == 1 / 2 * 2 * 5**2 + 1
    assert len(val_ds) == floor(1 / 4 * 2 * 5**2)
    assert len(test_ds) == floor(1 / 4 * 2 * 5**2)

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds
    assert isclose(total_area(train_ds | val_ds | test_ds), total_area(ds))

    # Test __getitem__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)

    # Test invalid input fractions
    with pytest.raises(ValueError, match='Sum of input fractions must equal 1.'):
        random_grid_cell_assignment(ds, fractions=[1 / 2, 1 / 3, 1 / 4])
    with pytest.raises(
        ValueError, match='All items in input fractions must be greater than 0.'
    ):
        random_grid_cell_assignment(ds, fractions=[1 / 2, 3 / 4, -1 / 4])
    with pytest.raises(ValueError, match='Input grid_size must be greater than 1.'):
        random_grid_cell_assignment(ds, fractions=[1 / 2, 1 / 4, 1 / 4], grid_size=1)


def test_roi_split() -> None:
    geometry = [
        shapely.box(0, 0, 1, 1),
        shapely.box(1, 0, 2, 1),
        shapely.box(2, 0, 3, 1),
        shapely.box(3, 0, 4, 1),
    ]
    ds = CustomGeoDataset(geometry=geometry)

    train_ds, val_ds, test_ds = roi_split(
        ds,
        rois=[
            shapely.box(0, 0, 2, 1),
            shapely.box(2, 0, 3.5, 1),
            shapely.box(3.5, 0, 4, 1),
        ],
    )

    # Check datasets lengths
    assert len(train_ds) == 3
    assert len(val_ds) == 3
    assert len(test_ds) == 1

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds
    assert isclose(total_area(train_ds | val_ds | test_ds), total_area(ds))

    # Test __getitem__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)

    # Test invalid input rois
    with pytest.raises(ValueError, match="ROIs in input rois can't overlap."):
        roi_split(ds, rois=[shapely.box(0, 0, 2, 1), shapely.box(1, 0, 3, 1)])


@pytest.mark.parametrize(
    'lengths,expected_lengths',
    [
        # List of timestamps
        (
            [
                pd.Interval(pd.Timestamp(2025, 4, 25), pd.Timestamp(2025, 4, 26, 6)),
                pd.Interval(pd.Timestamp(2025, 4, 26, 6), pd.Timestamp(2025, 4, 28)),
                pd.Interval(pd.Timestamp(2025, 4, 28), pd.Timestamp(2025, 4, 29)),
            ],
            [2, 2, 1],
        ),
        # List of lengths
        (
            [
                pd.Timedelta(1.5, unit='days'),
                pd.Timedelta(1.5, unit='days'),
                pd.Timedelta(1, unit='day'),
            ],
            [2, 2, 1],
        ),
        # List of fractions (with remainder)
        ([1 / 2, 3 / 8, 1 / 8], [2, 2, 1]),
    ],
)
def test_time_series_split(
    lengths: Sequence[tuple[int, int] | int | float], expected_lengths: Sequence[int]
) -> None:
    geometry = [
        shapely.box(0, 0, 1, 1),
        shapely.box(0, 0, 1, 1),
        shapely.box(0, 0, 1, 1),
        shapely.box(0, 0, 1, 1),
    ]
    index = pd.IntervalIndex.from_tuples(
        [
            (datetime(2025, 4, 25), datetime(2025, 4, 26)),
            (datetime(2025, 4, 26), datetime(2025, 4, 27)),
            (datetime(2025, 4, 27), datetime(2025, 4, 28)),
            (datetime(2025, 4, 28), datetime(2025, 4, 29)),
        ],
        closed='neither',
        name='datetime',
    )
    ds = CustomGeoDataset(index, geometry)

    train_ds, val_ds, test_ds = time_series_split(ds, lengths)

    # Check datasets lengths
    assert len(train_ds) == expected_lengths[0]
    assert len(val_ds) == expected_lengths[1]
    assert len(test_ds) == expected_lengths[2]

    print(train_ds.index)
    print(val_ds.index)

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds

    # Test __getitem__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)


def test_time_series_split_invalid_input() -> None:
    with pytest.raises(
        ValueError,
        match="Pairs of timestamps in lengths must cover dataset's time bounds.",
    ):
        time_series_split(
            CustomGeoDataset(),
            lengths=[
                pd.Interval(pd.Timestamp(2025, 4, 24, 0), pd.Timestamp(2025, 4, 24, 6)),
                pd.Interval(
                    pd.Timestamp(2025, 4, 24, 6), pd.Timestamp(2025, 4, 24, 12)
                ),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Pairs of timestamps in lengths can't be out of dataset's time bounds.",
    ):
        time_series_split(
            CustomGeoDataset(),
            lengths=[
                pd.Interval(pd.Timestamp(2025, 4, 24), pd.Timestamp(2025, 4, 25)),
                pd.Interval(pd.Timestamp(2025, 4, 25), pd.Timestamp(2025, 4, 26)),
            ],
        )

    with pytest.raises(
        ValueError, match="Pairs of timestamps in lengths can't overlap."
    ):
        time_series_split(
            CustomGeoDataset(),
            lengths=[
                pd.Interval(
                    pd.Timestamp(2025, 4, 24, 0), pd.Timestamp(2025, 4, 24, 18)
                ),
                pd.Interval(pd.Timestamp(2025, 4, 24, 12), pd.Timestamp(2025, 4, 25)),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Sum of input lengths must equal 1 or the dataset's time length.",
    ):
        time_series_split(CustomGeoDataset(), lengths=[1 / 2, 1 / 2, 1 / 2])

    with pytest.raises(
        ValueError, match='All items in input lengths must be greater than 0.'
    ):
        time_series_split(CustomGeoDataset(), lengths=[20, 25, -5])
