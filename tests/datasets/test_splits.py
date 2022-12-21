# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from math import floor

import pytest
import torch
from torch.utils.data import TensorDataset

from torchgeo.datasets.splits import (  # random_bbox_splitting,; roi_split,
    random_bbox_assignment,
    random_nongeo_split,
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

    num_bbox = len(ds.index.count(ds.index.bounds))

    # Test list of lengths
    train_ds, val_ds, test_ds = random_bbox_assignment(ds, lengths=[2, 1, 1])
    assert len(train_ds.index.count(train_ds.index.bounds)) == 2
    assert len(val_ds.index.count(val_ds.index.bounds)) == 1
    assert len(test_ds.index.count(test_ds.index.bounds)) == 1

    # Test list of fractions
    train_ds, val_ds, test_ds = random_bbox_assignment(
        ds, lengths=[1 / 2, 1 / 4, 1 / 4]
    )
    assert len(train_ds.index.count(train_ds.index.bounds)) == num_bbox / 2
    assert len(val_ds.index.count(val_ds.index.bounds)) == num_bbox / 4
    assert len(test_ds.index.count(test_ds.index.bounds)) == num_bbox / 4

    # Test list of fractions with remainder
    train_ds, val_ds, test_ds = random_bbox_assignment(
        ds, lengths=[1 / 3, 1 / 3, 1 / 3]
    )
    assert len(train_ds.index.count(train_ds.index.bounds)) == floor(num_bbox / 3) + 1
    assert len(val_ds.index.count(val_ds.index.bounds)) == floor(num_bbox / 3)
    assert len(test_ds.index.count(test_ds.index.bounds)) == floor(num_bbox / 3)

    # Test invalid input lenghts
    with pytest.raises(
        ValueError,
        match="Sum of input lengths must equal 1 or the length of dataset's index.",
    ):
        random_bbox_assignment(ds, lengths=[2, 2, 1])
    with pytest.raises(
        ValueError, match="All items in input lengths must be greater than 0."
    ):
        random_bbox_assignment(ds, lengths=[1 / 2, 3 / 4, -1 / 4])
