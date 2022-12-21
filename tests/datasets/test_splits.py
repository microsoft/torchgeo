# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from math import floor

import pytest
import torch
from torch.utils.data import TensorDataset

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.splits import (
    random_bbox_assignment,
    random_bbox_splitting,
    random_nongeo_split,
    roi_split,
)
from torchgeo.datasets.utils import BoundingBox

from .test_geo import CustomGeoDataset


def test_random_nongeo_split() -> None:
    num_samples = 24
    x = torch.ones(num_samples, 5)
    y = torch.randint(low=0, high=2, size=(num_samples,))
    ds = TensorDataset(x, y)

    # Test only train/val set split
    train_ds, val_ds = random_nongeo_split(ds, lengths=[1 / 2, 1 / 2])
    assert len(train_ds) == round(num_samples / 2)
    assert len(val_ds) == round(num_samples / 2)

    # Test train/val/test set split
    train_ds, val_ds, test_ds = random_nongeo_split(ds, lengths=[1 / 3, 1 / 3, 1 / 3])
    assert len(train_ds) == round(num_samples / 3)
    assert len(val_ds) == round(num_samples / 3)
    assert len(test_ds) == round(num_samples / 3)


def test_random_bbox_assignment() -> None:
    ds = (
        CustomGeoDataset(BoundingBox(0, 1, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(1, 2, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(2, 3, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(3, 4, 0, 1, 0, 0))
    )

    # Test list of lengths
    train_ds, val_ds, test_ds = random_bbox_assignment(ds, lengths=[2, 1, 1])
    assert len(train_ds) == 2
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert len(train_ds & val_ds & test_ds) == 0
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds

    # Test list of fractions (with remainder)
    train_ds, val_ds, test_ds = random_bbox_assignment(
        ds, lengths=[1 / 3, 1 / 3, 1 / 3]
    )
    assert len(train_ds) == floor(len(ds) / 3) + 1
    assert len(val_ds) == floor(len(ds) / 3)
    assert len(test_ds) == floor(len(ds) / 3)
    assert len(train_ds & val_ds & test_ds) == 0
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds

    # Test invalid input lengths
    with pytest.raises(
        ValueError,
        match="Sum of input lengths must equal 1 or the length of dataset's index.",
    ):
        random_bbox_assignment(ds, lengths=[2, 2, 1])
    with pytest.raises(
        ValueError, match="All items in input lengths must be greater than 0."
    ):
        random_bbox_assignment(ds, lengths=[1 / 2, 3 / 4, -1 / 4])


def get_total_area(dataset: GeoDataset) -> float:

    total_area = 0.0
    for hit in dataset.index.intersection(dataset.index.bounds, objects=True):
        total_area += BoundingBox(*hit.bounds).area

    return total_area


def test_random_bbox_splitting() -> None:
    ds = (
        CustomGeoDataset(BoundingBox(0, 1, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(1, 2, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(2, 3, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(3, 4, 0, 1, 0, 0))
    )

    ds_area = get_total_area(ds)

    # Test list of fractions
    train_ds, val_ds, test_ds = random_bbox_splitting(
        ds, fractions=[1 / 2, 1 / 4, 1 / 4]
    )
    train_ds_area = get_total_area(train_ds)
    val_ds_area = get_total_area(val_ds)
    test_ds_area = get_total_area(test_ds)

    assert train_ds_area == ds_area / 2
    assert val_ds_area == ds_area / 4
    assert test_ds_area == ds_area / 4
    assert len(train_ds & val_ds & test_ds) == 0
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds

    # Test invalid input fractions
    with pytest.raises(ValueError, match="Sum of input fractions must equal 1."):
        random_bbox_splitting(ds, fractions=[1 / 2, 1 / 3, 1 / 4])
    with pytest.raises(
        ValueError, match="All items in input lengths must be greater than 0."
    ):
        random_bbox_splitting(ds, fractions=[1 / 2, 3 / 4, -1 / 4])


def test_roi_split() -> None:
    ds = (
        CustomGeoDataset(BoundingBox(0, 1, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(1, 2, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(2, 3, 0, 1, 0, 0))
        | CustomGeoDataset(BoundingBox(3, 4, 0, 1, 0, 0))
    )

    train_ds, val_ds, test_ds = roi_split(
        ds,
        rois=[
            BoundingBox(0, 2, 0, 1, 0, 0),
            BoundingBox(2, 3.5, 0, 1, 0, 0),
            BoundingBox(3.5, 4, 0, 1, 0, 0),
        ],
    )
    assert len(train_ds) == 2
    assert len(val_ds) == 2
    assert len(test_ds) == 1
    assert len(train_ds & val_ds & test_ds) == 0
    assert (train_ds | val_ds | test_ds).bounds == ds.bounds

    # Test invalid input rois
    with pytest.raises(ValueError, match="ROIs in input roi should not overlap."):
        roi_split(
            ds, rois=[BoundingBox(0, 2, 0, 1, 0, 0), BoundingBox(1, 3, 0, 1, 0, 0)]
        )
