# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Sequence
from math import floor, isclose

import pytest
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    GeoDataset,
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)
from torchgeo.datasets.utils import Sample


def total_area(dataset: GeoDataset) -> float:
    total_area = 0.0
    for hit in dataset.index.intersection(dataset.index.bounds, objects=True):
        total_area += BoundingBox(*hit.bounds).area

    return total_area


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
        items: list[tuple[BoundingBox, str]] = [(BoundingBox(0, 1, 0, 1, 0, 40), '')],
        crs: CRS = CRS.from_epsg(3005),
        res: float = 1,
    ) -> None:
        super().__init__()
        for box, content in items:
            self.index.insert(0, tuple(box), content)
        self._crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> Sample:
        hits = self.index.intersection(tuple(query), objects=True)
        hit = next(iter(hits))
        return {'content': hit.object}


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
    ds = CustomGeoDataset(
        [
            (BoundingBox(0, 1, 0, 1, 0, 0), 'a'),
            (BoundingBox(1, 2, 0, 1, 0, 0), 'b'),
            (BoundingBox(2, 3, 0, 1, 0, 0), 'c'),
            (BoundingBox(3, 4, 0, 1, 0, 0), 'd'),
        ]
    )

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
    assert isinstance(x['content'], str)


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
    ds = CustomGeoDataset(
        [
            (BoundingBox(0, 1, 0, 1, 0, 0), 'a'),
            (BoundingBox(1, 2, 0, 1, 0, 0), 'b'),
            (BoundingBox(2, 3, 0, 1, 0, 0), 'c'),
            (BoundingBox(3, 4, 0, 1, 0, 0), 'd'),
        ]
    )

    ds_area = total_area(ds)

    train_ds, val_ds, test_ds = random_bbox_splitting(
        ds, fractions=[1 / 2, 1 / 4, 1 / 4]
    )
    train_ds_area = total_area(train_ds)
    val_ds_area = total_area(val_ds)
    test_ds_area = total_area(test_ds)

    # Check datasets areas
    assert train_ds_area == ds_area / 2
    assert val_ds_area == ds_area / 4
    assert test_ds_area == ds_area / 4

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds
    assert isclose(total_area(train_ds | val_ds | test_ds), ds_area)

    # Test __get_item__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)
    assert isinstance(x['content'], str)

    # Test invalid input fractions
    with pytest.raises(ValueError, match='Sum of input fractions must equal 1.'):
        random_bbox_splitting(ds, fractions=[1 / 2, 1 / 3, 1 / 4])
    with pytest.raises(
        ValueError, match='All items in input fractions must be greater than 0.'
    ):
        random_bbox_splitting(ds, fractions=[1 / 2, 3 / 4, -1 / 4])


def test_random_grid_cell_assignment() -> None:
    ds = CustomGeoDataset(
        [
            (BoundingBox(0, 12, 0, 12, 0, 0), 'a'),
            (BoundingBox(12, 24, 0, 12, 0, 0), 'b'),
        ]
    )

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

    # Test __get_item__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)
    assert isinstance(x['content'], str)

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
    ds = CustomGeoDataset(
        [
            (BoundingBox(0, 1, 0, 1, 0, 0), 'a'),
            (BoundingBox(1, 2, 0, 1, 0, 0), 'b'),
            (BoundingBox(2, 3, 0, 1, 0, 0), 'c'),
            (BoundingBox(3, 4, 0, 1, 0, 0), 'd'),
        ]
    )

    train_ds, val_ds, test_ds = roi_split(
        ds,
        rois=[
            BoundingBox(0, 2, 0, 1, 0, 0),
            BoundingBox(2, 3.5, 0, 1, 0, 0),
            BoundingBox(3.5, 4, 0, 1, 0, 0),
        ],
    )

    # Check datasets lengths
    assert len(train_ds) == 2
    assert len(val_ds) == 2
    assert len(test_ds) == 1

    # No overlap
    assert no_overlap(train_ds, val_ds)
    assert no_overlap(val_ds, test_ds)
    assert no_overlap(test_ds, train_ds)

    # Union equals original
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds
    assert isclose(total_area(train_ds | val_ds | test_ds), total_area(ds))

    # Test __get_item__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)
    assert isinstance(x['content'], str)

    # Test invalid input rois
    with pytest.raises(ValueError, match="ROIs in input rois can't overlap."):
        roi_split(
            ds, rois=[BoundingBox(0, 2, 0, 1, 0, 0), BoundingBox(1, 3, 0, 1, 0, 0)]
        )


@pytest.mark.parametrize(
    'lengths,expected_lengths',
    [
        # List of timestamps
        ([(0, 20), (20, 35), (35, 40)], [2, 2, 1]),
        # List of lengths
        ([20, 15, 5], [2, 2, 1]),
        # List of fractions (with remainder)
        ([1 / 2, 3 / 8, 1 / 8], [2, 2, 1]),
    ],
)
def test_time_series_split(
    lengths: Sequence[tuple[int, int] | int | float], expected_lengths: Sequence[int]
) -> None:
    ds = CustomGeoDataset(
        [
            (BoundingBox(0, 1, 0, 1, 0, 10), 'a'),
            (BoundingBox(0, 1, 0, 1, 10, 20), 'b'),
            (BoundingBox(0, 1, 0, 1, 20, 30), 'c'),
            (BoundingBox(0, 1, 0, 1, 30, 40), 'd'),
        ]
    )

    train_ds, val_ds, test_ds = time_series_split(ds, lengths)

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

    # Test __get_item__
    x = train_ds[train_ds.bounds]
    assert isinstance(x, dict)
    assert isinstance(x['content'], str)


def test_time_series_split_invalid_input() -> None:
    with pytest.raises(
        ValueError,
        match='Pairs of timestamps in lengths must have end greater than start.',
    ):
        time_series_split(CustomGeoDataset(), lengths=[(0, 20), (35, 20), (35, 40)])

    with pytest.raises(
        ValueError,
        match="Pairs of timestamps in lengths must cover dataset's time bounds.",
    ):
        time_series_split(CustomGeoDataset(), lengths=[(0, 20), (20, 35)])

    with pytest.raises(
        ValueError,
        match="Pairs of timestamps in lengths can't be out of dataset's time bounds.",
    ):
        time_series_split(CustomGeoDataset(), lengths=[(0, 20), (20, 45)])

    with pytest.raises(
        ValueError, match="Pairs of timestamps in lengths can't overlap."
    ):
        time_series_split(CustomGeoDataset(), lengths=[(0, 10), (10, 20), (15, 40)])

    with pytest.raises(
        ValueError,
        match="Sum of input lengths must equal 1 or the dataset's time length.",
    ):
        time_series_split(CustomGeoDataset(), lengths=[1 / 2, 1 / 2, 1 / 2])

    with pytest.raises(
        ValueError, match='All items in input lengths must be greater than 0.'
    ):
        time_series_split(CustomGeoDataset(), lengths=[20, 25, -5])
